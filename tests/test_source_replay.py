"""Numerical objectives, source routing, paired storage and exact resume tests."""
from pathlib import Path
import copy
import json
import sys
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dasr_aligned import continual_baselines as host
from dasr_aligned.baseline_losses import BaselineOptions
from dasr_aligned.baseline_memory import ReplayMemory
from dasr_aligned.common import ExperimentConfig, DataBundle, EMA, config_to_dict
from dasr_aligned.source_replay import runner as gsa
from dasr_aligned.source_replay.objective import GradientStatistics
from dasr_aligned.source_replay.derpp_memory import DERSourceMemory, initialize_from_shared_base
from dasr_aligned.source_replay.routing import regenerate_history
from dasr_aligned.source_replay.memory import HistoricalMemory

HAS_SHARED = 'shared_clean' in ExperimentConfig.__dataclass_fields__

@pytest.fixture(autouse=True)
def threads():
    n = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(n)


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else:
        assert a == b


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
        self.drop = torch.nn.Dropout(.2)
        self.fc = torch.nn.Linear(4, 10)
    def forward(self, x, return_feature=False):
        f = self.bn(self.conv(x)).relu().mean((2, 3))
        y = self.fc(self.drop(f))
        return (y, f, f, f) if return_feature else y


def config(path, method='derpp', reverse=False, shared=False):
    extra = {'shared_clean': shared} if HAS_SHARED else {}
    cfg = ExperimentConfig(method=method, dataset='cifar10', device='cpu', workers=0,
        backbone='resnet18', batch_size=2, output_root=str(path), run_name='test',
        task_epochs=dict.fromkeys(host.TASKS, 2), train_attack_steps=1, eval_attack_steps=1,
        cw_eval_steps=1, patch_size=2, final_test_methods=('clean','patch','pgd_l2','pgd'), **extra)
    cfg.task_sequence = ('clean','pgd','pgd_l2','patch') if reverse else tuple(host.TASKS)
    return cfg


def bundle():
    rng = torch.Generator().manual_seed(72)
    ds = TensorDataset(torch.rand(4,3,32,32,generator=rng), torch.arange(4))
    val_ds = TensorDataset(torch.rand(4,3,32,32,generator=rng), torch.arange(4))
    loader = DataLoader(ds, batch_size=2)
    val = DataLoader(val_ds, batch_size=2)
    probe = DataLoader(torch.utils.data.Subset(val_ds,[0]), batch_size=1)
    return DataBundle(train=loader, train_eval=loader, val=val, selection_val=probe,
                      test=val, train_indices=list(range(4)), val_indices=list(range(4,8)))


def runtime(monkeypatch):
    data = bundle()
    for mod in (host, gsa):
        monkeypatch.setattr(mod, 'build_dataloaders', lambda cfg: data)
        monkeypatch.setattr(mod, 'make_backbone', lambda cfg: Tiny())
        actual = mod.evaluate_seen
        def full(model, loader, tasks, cfg, device, actual=actual):
            assert loader is data.val and len(loader.dataset) == 4
            return actual(model, loader, tasks, cfg, device)
        monkeypatch.setattr(mod, 'evaluate_seen', full)
    return data


def test_gsa_upstream_weight_loss_gradient():
    torch.manual_seed(19)
    statistics = GradientStatistics(4, warmup_steps=0)
    counts, pos, neg = [torch.zeros(4,dtype=torch.float64) for _ in range(3)]
    for step in range(3):
        y = torch.tensor([0,0,1,2,3,0,2])
        z = torch.randn(7,4,dtype=torch.float64,requires_grad=True)
        ref = z.detach().clone().requires_grad_(True)
        p = ref.softmax(1)
        for i, label in enumerate(y):
            counts[label] += 1
            pos[label] += p[i,label].detach()
            for c in range(4):
                if c != label: neg[c] += p[i,c].detach()
        rate = (counts-pos)/neg
        weights = 2/(1+torch.exp(1-rate))
        expected = sum(-p[i,label].log()*weights[label] for i,label in enumerate(y))/len(y)
        actual = statistics.loss(z,y,stage=step//2)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        actual.backward(); expected.backward()
        torch.testing.assert_close(z.grad,ref.grad,atol=1e-12,rtol=1e-12)
        assert not statistics.weights().requires_grad


def test_gsa_single_sample_mean_no_duplicate_current_loss():
    stats = GradientStatistics(3)
    z = torch.tensor([[4.,0.,0.],[0.,3.,0.],[0.,0.,2.]],requires_grad=True)
    y = torch.tensor([0,2,1])
    actual = stats.loss(z,y,0)
    expected = F.cross_entropy(z,y)
    torch.testing.assert_close(actual,expected)
    actual.backward()
    assert int(stats.total_updates) == 1 and stats.category_count.sum() == 3


def test_gsa_warmup_extremes_and_roundtrip():
    stats = GradientStatistics(4,warmup_steps=2)
    z = torch.tensor([[10000.,-10000.,-10000.,-10000.]],requires_grad=True)
    y = torch.tensor([1])
    for i in range(3):
        loss = stats.loss(z,y,0)
        assert torch.isfinite(loss)
        if i<2: assert torch.equal(stats.weights(),torch.ones(4,dtype=torch.float64))
    loss.backward(); assert torch.isfinite(z.grad).all()
    saved = copy.deepcopy(stats.state_dict())
    new = GradientStatistics(4,warmup_steps=2); new.load_state_dict(saved)
    equal(stats.loss(z,y,1),new.loss(z,y,1)); equal(stats.state_dict(),new.state_dict())


@pytest.mark.parametrize('tasks', [('clean','patch','pgd_l2','pgd'),('clean','pgd','pgd_l2','patch'),
                                  ('pgd_l2','stadv','pgd','recolor')])
def test_source_routing_uses_sequence_current_model_and_host_cfg(tasks):
    cfg = config(Path('unused'))
    model = Tiny(); model.train()
    replay = {'images':torch.rand(6,3,4,4), 'labels':torch.arange(6),
              'tasks':torch.tensor([2,0,1,2,1,0])}
    calls=[]
    def attack(m,x,y,task,seen_cfg,training,rng_seed):
        assert m is model and seen_cfg is cfg and training
        calls.append((task,len(y),rng_seed))
        m.eval()
        return x+tasks.index(task)+1
    output, counts, generated = regenerate_history(model,replay,tasks,3,2,4,cfg,attack)
    assert model.training
    for i,stage in enumerate(replay['tasks']):
        expected = replay['images'][i] if tasks[int(stage)]=='clean' else replay['images'][i]+int(stage)+1
        torch.testing.assert_close(output[i],expected)
    assert {x[0] for x in calls} == {tasks[i] for i in (0,1,2) if tasks[i]!='clean'}
    assert generated == sum(r['generated'] for r in counts.values())
    assert not output.requires_grad


def test_current_stage_only_allowed_for_host_der_reservoir():
    cfg=config(Path('unused')); rep={'images':torch.zeros(1,3,2,2),'labels':torch.tensor([0]),'tasks':torch.tensor([1])}
    with pytest.raises(ValueError): regenerate_history(Tiny(),rep,host.TASKS,1,0,0,cfg)
    calls=[]
    regenerate_history(Tiny(),rep,host.TASKS,1,0,0,cfg,lambda m,x,y,t,*a,**k: calls.append(t) or x,allow_current=True)
    assert calls==['patch']
    rep['tasks']=torch.tensor([2])
    with pytest.raises(ValueError): regenerate_history(Tiny(),rep,host.TASKS,1,0,0,cfg,allow_current=True)


def test_der_memory_pairing_same_reservoir_and_resume():
    m=DERSourceMemory(3,(3,2,2),10,77,host.TASKS)
    old=ReplayMemory(3,(3,2,2),10,True,77)
    clean=torch.arange(8.).view(8,1,1,1).expand(8,3,2,2)
    anchors=clean+.25; labels=torch.arange(8); logits=labels[:,None].expand(8,10).float()
    m.add(anchors,labels,1,logits,clean_images=clean); old.add(anchors,labels,1,logits)
    state=m.state_dict()
    for key,val in old.state_dict().items(): equal(state[key],val)
    new=DERSourceMemory(3,(3,2,2),10,999,host.TASKS); new.load_state_dict(state)
    a=m.sample(3,'cpu');b=new.sample(3,'cpu');equal(a,b)
    for i,label in enumerate(a['labels']):
        assert torch.equal(a['clean_images'][i],clean[label])
        assert torch.equal(a['images'][i],anchors[label])
        assert torch.equal(a['logits'][i],logits[label])
    assert m.summary()['cpu_allocated_bytes']==m.summary()['valid_storage_bytes']
    with pytest.raises(ValueError): new.load_state_dict(old.state_dict())
    with pytest.raises(ValueError): m.add(anchors,labels,1,logits)


def test_der_manual_objective_preserves_anchor_mse(monkeypatch):
    cfg=config(Path('unused'));cfg.max_train_batches=1
    torch.manual_seed(18)
    model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(12,10))
    ref=copy.deepcopy(model)
    options=BaselineOptions('derpp',buffer_size=4,grad_clip=0,der_alpha=.3,der_beta=.8)
    memory=DERSourceMemory(4,(3,2,2),10,27,cfg.task_sequence)
    x=torch.rand(2,3,2,2);y=torch.tensor([0,1]);seeds=torch.rand(4,3,2,2)
    targets=torch.randn(4,10); labs=torch.tensor([2,3,4,5])
    memory.add(seeds[:2]+.2,labs[:2],1,targets[:2],seeds[:2])
    memory.add(seeds[2:]+.2,labs[2:],2,targets[2:],seeds[2:])
    sample=copy.deepcopy(memory).sample(2,'cpu')
    calls=[]
    shift={'clean':0.,'patch':.01,'pgd_l2':.02,'pgd':.03}
    def attack(m,images,labels,task,*a,**kw):
        calls.append(task); return (images+shift[task]).detach()
    monkeypatch.setattr(host,'generate_attack',attack)
    current=ref(x+.03)
    mse=F.mse_loss(ref(sample['images']),sample['logits'])
    fresh=torch.stack([im+shift[cfg.task_sequence[int(t)]] for im,t in zip(sample['clean_images'],sample['tasks'])])
    ce=F.cross_entropy(ref(fresh),sample['labels'])
    expected=F.cross_entropy(current,y)+.3*mse+.8*ce
    optref=torch.optim.SGD(ref.parameters(),lr=.01);expected.backward();optref.step()
    opt=torch.optim.SGD(model.parameters(),lr=.01)
    stats=host.train_epoch(model,EMA(model,.9),DataLoader(TensorDataset(x,y),batch_size=2),opt,
                           'pgd',3,0,cfg,options,memory,torch.device('cpu'))
    assert stats['loss']==pytest.approx(float(expected.detach()),abs=1e-6)
    assert stats['logit_mse']==pytest.approx(float(mse.detach()),abs=1e-6)
    for a,b in zip(model.parameters(),ref.parameters()): torch.testing.assert_close(a,b)
    assert calls[0]=='pgd' and set(calls[1:]).issubset({'patch','pgd_l2'})
    assert stats['generated_replay']==2


@pytest.mark.parametrize('task',['clean','pgd_l2'])
def test_shared_base_der_pair_alignment(task,monkeypatch):
    cfg=config(Path('unused'));cfg.task_sequence=(task,'patch','pgd','recolor')
    data=bundle(); model=Tiny();model.train()
    m=DERSourceMemory(8,(3,32,32),10,77,cfg.task_sequence)
    initialize_from_shared_base(m,model,data,cfg,torch.device('cpu'))
    assert model.training and len(m)==4
    x,y=next(iter(data.train_eval))
    assert torch.equal(m.clean_images[:2],x)
    model.eval()
    with torch.no_grad(): torch.testing.assert_close(model(m.images[:4]),m.logits[:4])
    if task=='clean': assert torch.equal(m.images[:4],m.clean_images[:4])
    else: assert not torch.equal(m.images[:4],m.clean_images[:4])


@pytest.mark.parametrize('method',['derpp','gsa'])
@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('interrupt_stage,interrupt_epoch',[(0,1),(2,2)])
def test_four_stage_training_exact_resume(method,reverse,interrupt_stage,interrupt_epoch,tmp_path,monkeypatch):
    runtime(monkeypatch)
    cfg=config(tmp_path/'full',method,reverse);cfg2=config(tmp_path/'resume',method,reverse)
    options=BaselineOptions(method,buffer_size=5)
    expected=host.run(cfg,options)
    mod=gsa if method=='gsa' else host
    actual_save=mod.atomic_checkpoint
    def interrupt(path,value):
        actual_save(path,value)
        if Path(path).name=='last.pt' and value['stage']==interrupt_stage and value['next_epoch']==interrupt_epoch:
            raise KeyboardInterrupt('test interruption')
    monkeypatch.setattr(mod,'atomic_checkpoint',interrupt)
    with pytest.raises(KeyboardInterrupt): host.run(cfg2,options)
    monkeypatch.setattr(mod,'atomic_checkpoint',actual_save)
    cfg2.resume=True
    assert host.run(cfg2,options)==expected
    a=torch.load(cfg.run_dir/'last.pt',weights_only=False,map_location='cpu')
    b=torch.load(cfg2.run_dir/'last.pt',weights_only=False,map_location='cpu')
    for name in ('model','ema','optimizer','scheduler','memory','best_snapshot','epoch_rows','best_score','stage_summaries'):
        equal(a[name],b[name])
    if method=='gsa':
        equal(a['gsa_statistics'],b['gsa_statistics'])
        assert a['memory']['num_seen']==16 # one 4-image pass per completed stage
        assert a['memory']['last_committed_stage']==3
        for row in a['epoch_rows']:
            assert row['gsa_updates']==2
            assert all(int(s)<row['stage'] for s in row['replay_by_source'])
    else:
        assert a['memory']['num_seen']==32 # host stream: 4 stages * 2 epochs * 4 images
        assert 'clean_images' in a['memory']
    monkeypatch.setattr(mod,'train_epoch',lambda *a,**kw: pytest.fail('Unexpected training during eval-only'))
    assert host.run(cfg2,options,eval_only=True)==expected


@pytest.mark.parametrize('method',['derpp','gsa'])
def test_old_checkpoint_rejected_before_writes(method,tmp_path,monkeypatch):
    cfg=config(tmp_path,method);cfg.run_dir.mkdir(parents=True)
    marker=cfg.run_dir/'baseline_config.json';marker.write_text('OLD')
    torch.save({'signature':'old-original-method'},cfg.run_dir/'last.pt')
    cfg.resume=True
    with pytest.raises(ValueError,match='mismatch'): host.run(cfg,BaselineOptions(method,buffer_size=5))
    assert marker.read_text()=='OLD'


@pytest.mark.skipif(not HAS_SHARED,reason='This host has no shared-base feature')
@pytest.mark.parametrize('method',['derpp','gsa'])
def test_shared_base_mode_keeps_host_origin(method,tmp_path,monkeypatch):
    runtime(monkeypatch)
    from dasr_aligned import shared_base
    def acquire(cfg,loaders,model,device):
        return {'model':copy.deepcopy(model.state_dict()),'accuracy':50.,'epoch':0,
                'signature':'existing_shared_base','model_sha256':'unchanged_base'}
    monkeypatch.setattr(shared_base,'acquire',acquire)
    monkeypatch.setattr(shared_base,'record_import',lambda *a:None)
    cfg=config(tmp_path,method,shared=True)
    host.run(cfg,BaselineOptions(method,buffer_size=5))
    rows=json.loads((cfg.run_dir/'epoch_metrics.json').read_text())
    assert all(row['stage']>0 for row in rows)
    base=torch.load(cfg.run_dir/'stage0_clean_best.pt',map_location='cpu',weights_only=False)
    assert base['shared_base_model_sha256']=='unchanged_base'


@pytest.mark.parametrize('method',['derpp','gsa'])
def test_real_resnet18_imagenette_single_update(method):
    from dasr_aligned.common import make_backbone
    args=host.build_parser(method).parse_args(['--dataset','imagenette','--backbone','resnet18',
        '--device','cpu','--workers','0','--batch-size','2','--buffer-size','4',
        '--train-attack-steps','1','--max-train-batches','1'])
    cfg,options=host.resolve_config(method,args)
    model=make_backbone(cfg);optimizer=torch.optim.SGD(model.parameters(),lr=.001)
    data=TensorDataset(torch.rand(2,3,160,160),torch.tensor([0,1]));loader=DataLoader(data,batch_size=2)
    ema=EMA(model,cfg.ema_decay)
    if method=='derpp':
        memory=host.make_memory(cfg,options)
        x,y=next(iter(loader));memory.add(x,y,0,model(x).detach(),clean_images=x)
        result=host.train_epoch(model,ema,loader,optimizer,'patch',1,0,cfg,options,memory,torch.device('cpu'))
    else:
        memory=HistoricalMemory(4,(3,160,160),10,42);memory.commit_stage(loader,0,cfg,torch.device('cpu'))
        result=gsa.train_epoch(model,ema,loader,optimizer,'patch',1,0,cfg,gsa.from_host_options(options),
                               memory,GradientStatistics(10),torch.device('cpu'))
    assert result['optimizer_updates']==1 and result['replay_examples']==2

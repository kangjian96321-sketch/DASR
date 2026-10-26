"""Release-specific defaults, sequence dispatch, warmup and ALR integration."""
import copy
import importlib
import json
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dasr_aligned import common, continual_baselines, joint, crt, sseat
from dasr_aligned.selection import (OLD_SEQUENCE, NEW_SEQUENCE, SELECTION_POLICY,
                                    warmup_epochs, best_eligible, selection_mean)
from dasr_aligned.crt_lp_alr import official_lp_alr, generate_lp_alr_view
from check_crt_lp_alr import ProbeModel, ZeroInputGradient
import train_DASR as dasr


@pytest.mark.parametrize('epochs,warmup,first', [(100,6,7),(30,2,3),(3,1,2),(2,1,2),(1,0,1)])
def test_scheduler_and_best_share_warmup(epochs, warmup, first):
    cfg = common.ExperimentConfig(method='test', task_epochs={'clean': epochs})
    assert warmup_epochs(cfg, 'clean') == warmup
    assert [e + 1 for e in range(epochs) if best_eligible(cfg, 'clean', e)][0] == first
    optimizer = torch.optim.SGD(torch.nn.Linear(1,1).parameters(), lr=.1)
    scheduler = common.make_scheduler(optimizer, 'clean', cfg)
    if warmup:
        assert scheduler._milestones == [warmup]


def test_zero_inclusion_and_invalid_scores():
    assert selection_mean({'clean':90., 'patch':0.}, ('clean','patch')) == 45.
    assert selection_mean({'clean':0., 'patch':0.}, ('clean','patch')) == 0.
    for values in ({'clean': math.nan}, {'clean':math.inf}, {'clean':-1.}, {}, {'clean':101.}):
        with pytest.raises(ValueError):
            selection_mean(values, ('clean',))


@pytest.mark.parametrize('method', ['trades','mart','gsa','derpp'])
@pytest.mark.parametrize('tag,sequence', [('clean_patch_l2_linf',OLD_SEQUENCE),('clean_linf_l2_patch',NEW_SEQUENCE)])
def test_baseline_both_entry_orders(method, tag, sequence, capsys):
    module = importlib.import_module(f'train_{method}_{tag}')
    module.main(['--dry-run','--dataset','imagenette','--backbone','resnet18'])
    cfg = json.loads(capsys.readouterr().out)['config']
    assert cfg['task_sequence'] == list(sequence)
    assert cfg['selection_policy'] == SELECTION_POLICY
    assert cfg['selection_probe_size'] == 500
    with pytest.raises(SystemExit):
        module.main(['--task-sequence', *OLD_SEQUENCE])


@pytest.mark.parametrize('tag,sequence', [('clean_patch_l2_linf',OLD_SEQUENCE),('clean_linf_l2_patch',NEW_SEQUENCE)])
def test_joint_orders_are_conditions_not_stages(tag, sequence, capsys):
    module = importlib.import_module(f'train_joint_{tag}')
    module.main(['--dry-run'])
    value = json.loads(capsys.readouterr().out)
    assert value['joint']['attacks'] == list(sequence)
    assert value['config']['task_sequence'] == ['clean','joint']


def test_far_defaults_and_no_removed_baseline():
    args = dasr.parse_args([])
    assert args.far_monitor_size == args.far_history_size == 500
    root = Path(__file__).resolve().parents[1]
    assert not (root/'tools'/'restart_patch.py').exists()
    cfg = common.ExperimentConfig(method='dasr')
    assert not any(k.startswith('cad_') for k in vars(cfg))


@pytest.mark.parametrize('task', ['pgd_l2','pgd'])
@pytest.mark.parametrize('steps', [1,2,4])
@pytest.mark.parametrize('training', [False,True])
def test_official_alr_dispatch_gradients_bn_rng(task, steps, training):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = ProbeModel().train(training)
    expected = copy.deepcopy(model)
    cfg = common.ExperimentConfig(method='crt')
    x = torch.rand(4,3,4,4)
    rng = torch.get_rng_state().clone()
    before = int(model.layers[1].num_batches_tracked)
    actual = crt.compute_alr(model, x, task, cfg, steps, 19)
    target = official_lp_alr(expected, x, task, cfg, steps, 19)
    torch.testing.assert_close(actual, target, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert int(model.layers[1].num_batches_tracked) - before == (2*steps+2 if training else 0)
    assert all(p.grad is None for p in model.parameters())
    actual.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.training == training


def test_l2_zero_gradient_fallback():
    cfg = common.ExperimentConfig(method='crt')
    x = torch.full((3,3,4,4), .5)
    view = generate_lp_alr_view(ZeroInputGradient(), x, 'pgd_l2', cfg, 4, 31)
    assert torch.isfinite(view).all()
    assert (view-x).flatten(1).norm(dim=1).max() <= cfg.l2_eps + 1e-6


def test_dasr_epoch_resume_does_not_select_warmup_checkpoint(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg = common.ExperimentConfig(method='dasr_uniform', dataset='cifar10', device='cpu', workers=0,
        output_root=str(tmp_path), run_name='warmup_resume', depth=10, widen_factor=1,
        batch_size=4, task_sequence=('clean',), task_epochs={'clean':3},
        max_train_batches=1, max_eval_batches=1, final_test_methods=('clean',))
    save = dasr.atomic_checkpoint
    def interrupt(path, payload):
        save(path, payload)
        if Path(path).name == 'stage0_clean_resume.pt' and payload['epoch'] == 0:
            assert payload['best_snapshot'] is None
            raise RuntimeError('stop in warmup')
    monkeypatch.setattr(dasr,'atomic_checkpoint',interrupt)
    with pytest.raises(RuntimeError,match='stop in warmup'):
        dasr.run_experiment(cfg, policy='uniform', smoke=True)
    assert not (cfg.run_dir/'stage0_clean_selected.pt').exists()
    monkeypatch.setattr(dasr,'atomic_checkpoint',save)
    cfg.resume = True
    dasr.run_experiment(cfg, policy='uniform', smoke=True, resume_mode='epoch')
    selected = torch.load(cfg.run_dir/'stage0_clean_selected.pt',weights_only=False)
    assert selected['epoch'] >= 1


@pytest.mark.parametrize('method', ['trades','mart','gsa','derpp'])
def test_reverse_baseline_training_selects_after_warmup(method, tmp_path, monkeypatch):
    # Reuse the native-loss synthetic harness; no model/dataset downloads.
    from test_four_baselines import config, patch_runtime
    patch_runtime(monkeypatch)
    cfg = config(tmp_path, method)
    cfg.task_sequence = NEW_SEQUENCE
    options = continual_baselines.BaselineOptions(method, buffer_size=5)
    continual_baselines.run(cfg, options)
    summaries = json.loads((cfg.run_dir/'stage_summaries.json').read_text())
    assert [r['task'] for r in summaries] == list(NEW_SEQUENCE)
    assert all(r['best_epoch'] == 2 for r in summaries)
    cfg.resume = True
    monkeypatch.setattr(continual_baselines, 'train_epoch', lambda *a, **k: pytest.fail('Retrained'))
    continual_baselines.run(cfg, options)


@pytest.mark.parametrize('method', ['sseat','crt'])
def test_native_best_ignores_warmup_maximum_and_counts_zero(method, tmp_path, monkeypatch):
    module = sseat if method == 'sseat' else crt
    torch.set_num_threads(1)
    cfg = common.ExperimentConfig(method=method, device='cpu', workers=0,
        run_directory=str(tmp_path/'run'), task_sequence=('clean',), task_epochs={'clean':3},
        batch_size=2, depth=10, widen_factor=1, final_test_methods=('clean',),
        sseat_memory_size=4, sseat_uncertainty_views=2)
    data = torch.utils.data.TensorDataset(torch.rand(4,3,32,32), torch.arange(4))
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    bundle = common.DataBundle(loader,loader,loader,loader,loader,[0,1],[2,3])
    monkeypatch.setattr(module,'build_dataloaders',lambda cfg:bundle)
    monkeypatch.setattr(module,'make_backbone',lambda cfg:torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3072,10)))
    scores = iter([100.,0.,0.])
    monkeypatch.setattr(module,'evaluate_for_selection',lambda *a,**k:{'clean':next(scores)})
    monkeypatch.setattr(module,'final_evaluation',lambda *a,**k:{'clean':0.})
    if method == 'sseat':
        module.run(cfg)
    else:
        module.run(cfg, crt.CRTOptions())
    saved = torch.load(cfg.run_dir/'stage_00_clean_best.pt',weights_only=False)
    assert saved['epoch'] == 1 and saved['best_score'] == 0.
    before = (cfg.run_dir/'config.json').read_bytes()
    old = json.loads(before); old.pop('selection_policy')
    (cfg.run_dir/'config.json').write_text(json.dumps(old),encoding='utf-8')
    cfg.resume = True
    with pytest.raises(ValueError,match='policy mismatch'):
        common.prepare_run_directory(cfg)

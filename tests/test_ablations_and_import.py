"""CPU integration tests using small WRNs and generated legacy-format weights."""
import copy
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_DASR as training
import checkpoint_compat as compat
from ablation_config import ABLATIONS, AblationConfig, make_criterion, method_name
from dasr_components import PureFeatureDistillationLoss
from dasr_aligned.common import ExperimentConfig
from summarize_ablations import collect, aggregate


def cfg_at(root, ablation='full'):
    torch.set_num_threads(2)
    return ExperimentConfig(method=method_name('uniform', ablation), dataset='cifar10',
        output_root=str(root), run_name='test', device='cpu', depth=10, widen_factor=1,
        workers=0, batch_size=4, train_attack_steps=1, eval_attack_steps=1,
        max_train_batches=1, max_eval_batches=1, final_test_methods=('clean',),
        task_epochs={t:2 for t in compat.TASKS})


def run(cfg, **kwargs):
    return training.run_experiment(cfg, policy='uniform', smoke=True, **kwargs)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


@pytest.fixture(scope='module')
def finished(tmp_path_factory):
    cfg = cfg_at(tmp_path_factory.mktemp('native'))
    run(cfg)
    return cfg


def legacy_files(finished, destination, stages, kinds=('best','last')):
    destination.mkdir(parents=True)
    for stage in stages:
        task = compat.TASKS[stage]
        selected = load(finished.run_dir/f'stage{stage}_{task}_selected.pt')
        for kind in kinds:
            torch.save({'model_state_dict': selected['model'], 'epoch': 1,
                'best_historical_accs': {task:99.9}}, destination/f'{kind}_Task{stage}_{task.title()}.pth')
    return destination


@pytest.mark.parametrize('preset', ABLATIONS)
def test_all_presets_train_and_save_expected_switches(tmp_path, preset):
    ablation = AblationConfig(preset)
    cfg = cfg_at(tmp_path, preset)
    run(cfg, ablation=ablation)
    saved = load(cfg.run_dir/'stage3_pgd_resume.pt')
    assert (saved['queue'] is not None) == ablation.use_queue
    assert (saved['teacher'] is not None) == ablation.use_teacher
    rows = compat.json_read(cfg.run_dir/'epoch_metrics.json')
    if not ablation.replay:
        assert all(r['replay_examples'] == 0 for r in rows)
    if preset == 'only_cls':
        assert all(r['acd'] == 0 and r['loss'] == r['classification'] for r in rows)
    if preset == 'fixed_weights':
        assert all(r['weight'] == training.BASE_WEIGHTS[r['task']] for r in rows)
    if ablation.use_queue:
        # Current and replay share a single enqueue of the four clean keys.
        assert sum(saved['queue']['class_counts']) == 8
    if preset == 'no_distillation':
        assert saved['settings']['ablation']['key_source'] == 'student'


def test_pure_distillation_formula_and_detached_teacher():
    q = torch.tensor([[1.,0.],[1.,1.]], requires_grad=True)
    k = torch.tensor([[1.,0.],[0.,1.]], requires_grad=True)
    loss = PureFeatureDistillationLoss()(q,k,queue_features=torch.randn(30,2))
    assert loss.item() == pytest.approx((1-2**-.5)/2)
    loss.backward()
    assert k.grad is None and q.grad.abs().sum() > 0
    assert make_criterion(AblationConfig('fixed_temperature'),'patch').temp_pos == .1
    assert make_criterion(AblationConfig('fixed_temperature'),'patch').temp_neg == .1


def test_legacy_completed_boundaries_repair_and_continue(tmp_path, finished):
    source = legacy_files(finished, tmp_path/'old', [0,1])
    hashes = {p.name:compat.file_hash(p) for p in source.iterdir()}
    cfg = cfg_at(tmp_path/'new')
    run(cfg, resume_source=source)
    assert hashes == {p.name:compat.file_hash(p) for p in source.iterdir()}
    rows = compat.json_read(cfg.run_dir/'epoch_metrics.json')
    assert {r['stage'] for r in rows} == {2,3}
    resources = compat.json_read(cfg.run_dir/'resource_metrics.json')
    assert resources[0]['training_wall_time_seconds'] is None
    assert resources[-1]['training_wall_time_seconds'] > 0
    assert compat.json_read(cfg.run_dir/'run_complete.json')['imported_history'] is True
    assert (cfg.run_dir/'stage0_clean_complete.json').exists()
    refs = compat.json_read(cfg.run_dir/'replay_references.json')
    assert refs['accuracies']['clean'] != 99.9
    # Re-running the same external import command follows the new local progress.
    training_before = compat.file_hash(cfg.run_dir/'stage3_pgd_resume.pt')
    run(cfg, resume_source=source)
    assert compat.file_hash(cfg.run_dir/'stage3_pgd_resume.pt') == training_before


def test_best_only_needs_completion_declaration(tmp_path, finished):
    source = legacy_files(finished, tmp_path/'old', [0,1], ('best',))
    refs = compat.discover(source)
    with pytest.raises(ValueError, match='only a best'):
        compat.plan(refs)
    assert compat.plan(refs, completed_through='patch')[0] == 2
    cfg = cfg_at(tmp_path/'new')
    run(cfg, resume_source=source, completed_through='patch')
    assert (cfg.run_dir/'final_test.json').exists()


def test_only_final_checkpoint_runs_test_and_marks_missing_history(tmp_path, finished, monkeypatch):
    source = legacy_files(finished, tmp_path/'old', [3])
    monkeypatch.setattr(training,'train_epoch', lambda *a,**k: pytest.fail('Unexpected training'))
    cfg = cfg_at(tmp_path/'new')
    run(cfg, resume_source=source)
    matrix = compat.json_read(cfg.run_dir/'accuracy_matrix.json')
    assert matrix[0]['accuracies'] is None
    assert matrix[-1]['accuracies'] is not None
    assert not compat.json_read(cfg.run_dir/'run_complete.json')['history_complete']


def test_manual_task_restart_archives_later_outputs(tmp_path, finished, monkeypatch):
    cfg = cfg_at(tmp_path)
    shutil.copytree(finished.run_dir, cfg.run_dir)
    previous = load(cfg.run_dir/'stage0_clean_selected.pt')['model']
    real_train = training.train_epoch
    calls = []
    def inspect_start(model,*args,**kwargs):
        stage, epoch = args[7], args[8]
        calls.append((stage,epoch))
        if len(calls) == 1:
            assert (stage,epoch) == (1,0)
            assert all(torch.equal(model.state_dict()[k],v) for k,v in previous.items())
        return real_train(model,*args,**kwargs)
    monkeypatch.setattr(training,'train_epoch',inspect_start)
    cfg.resume = True
    run(cfg, start_task='patch')
    assert (3,1) in calls
    assert list((cfg.run_dir/'backups').rglob('stage3_pgd_selected.pt'))
    assert len(compat.json_read(cfg.run_dir/'accuracy_matrix.json')) == 4


def test_default_resume_restarts_latest_unfinished_task(tmp_path, monkeypatch):
    cfg = cfg_at(tmp_path)
    real_save = training.atomic_checkpoint
    def interrupt(path, payload):
        real_save(path,payload)
        if Path(path).name == 'stage1_patch_resume.pt' and payload['epoch'] == 0:
            raise KeyboardInterrupt()
    monkeypatch.setattr(training,'atomic_checkpoint',interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(cfg)
    monkeypatch.setattr(training,'atomic_checkpoint',real_save)
    real_train = training.train_epoch
    calls = []
    def record(model,*args,**kwargs):
        calls.append((args[7],args[8]))
        return real_train(model,*args,**kwargs)
    monkeypatch.setattr(training,'train_epoch',record)
    cfg.resume = True
    run(cfg)
    assert calls[0] == (1,0)


def test_final_evaluation_cache_survives_json_repair(tmp_path, finished, monkeypatch):
    cfg = cfg_at(tmp_path)
    shutil.copytree(finished.run_dir,cfg.run_dir)
    cfg.resume = True
    cfg.final_test_methods = ('clean','autoattack_l2','autoattack')
    import dasr_aligned.common as common
    calls = []
    def aa(model,loader,norm,*args):
        calls.append(norm)
        if norm == 'Linf' and calls.count('Linf') == 1:
            raise KeyboardInterrupt()
        return 31.5
    monkeypatch.setattr(common,'evaluate_autoattack',aa)
    with pytest.raises(KeyboardInterrupt):
        run(cfg)
    (cfg.run_dir/'stage3_pgd_complete.json').unlink()
    run(cfg)
    assert calls == ['L2','Linf','Linf']
    assert compat.json_read(cfg.run_dir/'final_test_meta.json')['complete']


def test_last_resume_uses_embedded_selected_model(tmp_path, finished, monkeypatch):
    cfg = cfg_at(tmp_path)
    cfg.run_dir.mkdir(parents=True)
    name = 'stage3_pgd_resume.pt'
    payload = load(finished.run_dir/name)
    # Deliberately make the raw final model different from its selected model.
    key = next(k for k in payload['model'] if k.endswith('weight'))
    payload['model'][key] = torch.zeros_like(payload['model'][key])
    torch.save(payload,cfg.run_dir/name)
    cfg.resume = True
    monkeypatch.setattr(training,'train_epoch',lambda *a,**k:pytest.fail('Unexpected training'))
    run(cfg)
    selected = load(cfg.run_dir/'stage3_pgd_selected.pt')
    assert torch.equal(selected['model'][key],payload['best_snapshot']['model'][key])


def test_ema_parameters_overlay_retains_buffers(finished):
    cfg = cfg_at(finished.run_dir/'unused')
    model = training.make_backbone(cfg)
    state = training.cpu_state(model)
    payload = {'model_state_dict':{'module.'+k:v for k,v in state.items()},
        'ema_state_dict':{'module.'+k:torch.ones_like(v)*.25 for k,v in model.named_parameters()}}
    restored = compat.validated_state(payload, model, use_ema=True)
    for name,_ in model.named_parameters():
        assert torch.all(restored[name] == .25)
    assert torch.equal(restored['mean'],state['mean'])
    payload['model_state_dict']['module.mean'] = state['mean']+1
    with pytest.raises(ValueError,match='normalization'):
        compat.validated_state(payload,model)


def test_ablation_identity_mismatch_rejected_before_outputs(tmp_path, finished):
    cfg = cfg_at(tmp_path,'no_queue')
    with pytest.raises(ValueError,match='configuration/policy mismatch'):
        run(cfg,ablation=AblationConfig('no_queue'),resume_source=finished.run_dir)
    assert not cfg.run_dir.exists()


def test_summary_keeps_imported_runs_individual(tmp_path, finished):
    cfg = cfg_at(tmp_path/'first')
    shutil.copytree(finished.run_dir,cfg.run_dir)
    cfg2 = cfg_at(tmp_path/'second')
    shutil.copytree(finished.run_dir,cfg2.run_dir)
    data = compat.json_read(cfg2.run_dir/'comparison_config.json')
    data['settings']['config']['seed'] = 43
    training.atomic_json(cfg2.run_dir/'comparison_config.json',data)
    rows = collect(tmp_path,include_smoke=True)
    assert len(aggregate(rows)) == 1
    data['imported_history'] = True
    training.atomic_json(cfg2.run_dir/'comparison_config.json',data)
    assert aggregate(collect(tmp_path,include_smoke=True)) == []
    data['imported_history'] = False
    data['settings']['ablation']['weight_floor'] = .3
    training.atomic_json(cfg2.run_dir/'comparison_config.json',data)
    assert aggregate(collect(tmp_path,include_smoke=True)) == []


def test_latest_task_uses_task_order_and_explicit_raw_filename(tmp_path, finished):
    source = legacy_files(finished,tmp_path/'old',[0,2])
    assert compat.plan(compat.discover(source))[0] == 3
    raw = tmp_path/'my_model.pth'
    torch.save(load(finished.run_dir/'stage3_pgd_selected.pt')['model'],raw)
    assert compat.inspect(raw) is None
    assert compat.inspect(raw,checkpoint_task='pgd').stage == 3


def test_batch_launcher_default_and_mapping(tmp_path):
    import run_ablations
    from argparse import Namespace
    args = Namespace(seeds=[42],ablations=['full','no_queue'],checkpoint_map=None,
        output_root=str(tmp_path),dataset='cifar100',run_name='test',data_root='./data',
        policy='dtr',resume=True)
    commands = run_ablations.build_commands(args,['--smoke'])
    assert len(commands) == 2 and '--resume' not in commands[0]
    mapping = tmp_path/'map.json'
    training.atomic_json(mapping,{'no_queue':{'path':'old/no_queue','completed_through':'patch'}})
    args.checkpoint_map = mapping
    commands = run_ablations.build_commands(args,[])
    assert commands[1][-4:] == ['--resume','old/no_queue','--completed-through','patch']
    args.seeds = [42,43]
    with pytest.raises(ValueError,match='multiple seeds'):
        run_ablations.build_commands(args,[])


def interrupted_fixture(cfg, monkeypatch):
    real_save = training.atomic_checkpoint
    def interrupt(path, payload):
        real_save(path,payload)
        if Path(path).name == 'stage1_patch_resume.pt' and payload['epoch'] == 0:
            raise KeyboardInterrupt()
    monkeypatch.setattr(training,'atomic_checkpoint',interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(cfg)
    monkeypatch.setattr(training,'atomic_checkpoint',real_save)


def test_native_epoch_resume_from_self_contained_checkpoint(tmp_path, monkeypatch):
    initial = cfg_at(tmp_path/'initial')
    interrupted_fixture(initial,monkeypatch)
    source = tmp_path/'only_resume'; source.mkdir()
    shutil.copy2(initial.run_dir/'stage1_patch_resume.pt',source)
    cfg = cfg_at(tmp_path/'continued')
    run(cfg,resume_source=source,resume_mode='epoch')
    result = load(cfg.run_dir/'stage1_patch_resume.pt')
    assert sum(result['queue']['class_counts']) == 8
    assert [r['epoch'] for r in result['epoch_rows']] == [0,1]
    assert result['reference_sources']['clean']['origin'] == 'native'
    assert (cfg.run_dir/'final_test.json').exists()


def test_legacy_epoch_resume_with_missing_queue_and_historical_cost(tmp_path, monkeypatch):
    initial = cfg_at(tmp_path/'initial')
    interrupted_fixture(initial,monkeypatch)
    source = legacy_files(initial,tmp_path/'old',[0],('best','last'))
    saved = load(initial.run_dir/'stage1_patch_resume.pt')
    legacy = {'model_state_dict':saved['model'],'ema_state_dict':saved['ema'],
        'optimizer_state_dict':saved['optimizer'],'scheduler_state_dict':saved['scheduler'],
        'epoch':saved['epoch'],'running_sim':saved['running_sim']}
    torch.save(legacy,source/'resume_Task1_Patch.pth')
    cfg = cfg_at(tmp_path/'continued')
    run(cfg,resume_source=source,resume_mode='epoch')
    result = load(cfg.run_dir/'stage1_patch_resume.pt')
    assert [r['epoch'] for r in result['epoch_rows']] == [1]
    assert sum(result['queue']['class_counts']) == 4
    resource = compat.json_read(cfg.run_dir/'resource_metrics.json')[1]
    assert resource['training_wall_time_seconds'] is None
    assert resource['continuation_measurements']['training_wall_time_seconds'] > 0
    assert result['recovery_notes']


def test_last_weights_do_not_invent_first_selected_reference(tmp_path, finished):
    source = legacy_files(finished,tmp_path/'old',[0],('last',))
    cfg = cfg_at(tmp_path/'new')
    with pytest.raises(ValueError,match='original best/selected'):
        run(cfg,resume_source=source)
    assert not (cfg.run_dir/'stage1_patch_resume.pt').exists()


def test_original_best_has_priority_over_converted_last(tmp_path, finished):
    source = legacy_files(finished,tmp_path/'old',[0],('best',))
    payload = load(finished.run_dir/'stage0_clean_selected.pt')
    payload['boundary_kind'] = 'last'
    payload['first_learned_reference'] = {}
    torch.save(payload,source/'stage0_clean_selected.pt')
    preferred = compat.preferred(compat.discover(source),0)
    assert preferred.kind == 'best' and preferred.selected_model

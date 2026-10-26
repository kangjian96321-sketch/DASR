"""Sequence-only regression. Tiny CPU runs use synthetic images, not real results."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_DASR as training
import train_DASR_clean_linf_l2_patch as entry
from far_replay import SOURCE_MODES
from dasr_aligned.common import ExperimentConfig, TASKS


NEW = ('clean', 'pgd', 'pgd_l2', 'patch')
OLD = ('clean', 'patch', 'pgd_l2', 'pgd')


@pytest.mark.parametrize('mode', SOURCE_MODES)
def test_new_entry_config_and_four_modes(mode, capsys, monkeypatch):
    monkeypatch.setattr(training, 'run_experiment', lambda *a, **k: pytest.fail('dry-run started training'))
    entry.main(['--dataset', 'imagenette', '--backbone', 'resnet18', '--batch-size', '32',
                '--far-monitor-size', '500', '--replay-source-mode', mode, '--dry-run'])
    result = json.loads(capsys.readouterr().out)
    assert result['task_sequence'] == list(NEW)
    cfg = result['config']
    assert cfg['task_sequence'] == list(NEW)
    assert cfg['task_epochs'] == {'clean': 100, 'patch': 30, 'pgd_l2': 30, 'pgd': 30}
    assert cfg['batch_size'] == 32 and cfg['patch_size'] == 40
    assert cfg['l2_eps'] == 2.5 and cfg['l2_alpha'] == .5
    assert cfg['linf_eps'] == 8 / 255 and cfg['train_attack_steps'] == 10
    assert cfg['eval_attack_steps'] == 20
    assert result['settings']['far_replay']['source_mode'] == mode
    assert result['run_dir'].startswith('runs_replay_clean_linf_l2_patch')


def test_original_defaults_and_joint_unchanged(capsys):
    training.main(['--dry-run'])
    result = json.loads(capsys.readouterr().out)
    assert result['task_sequence'] == list(OLD)
    assert TASKS == OLD
    from dasr_aligned import joint
    cfg, options = joint.resolve_config(joint.build_parser().parse_args([]))
    assert options.attacks == OLD and cfg.task_sequence == ('clean', 'joint')


@pytest.mark.parametrize('sequence', [
    ['clean', 'pgd', 'pgd', 'patch'], ['pgd', 'clean', 'pgd_l2', 'patch'],
    ['clean', 'pgd', 'patch'], ['clean', 'pgd', 'pgd_l2', 'stadv'],
])
def test_invalid_sequence_rejected(sequence):
    with pytest.raises(SystemExit):
        training.parse_args(['--task-sequence', *sequence])


def test_fixed_entry_rejects_conflicting_sequence():
    for flag in ('--task-sequence', '--task-seq'):
        with pytest.raises(SystemExit, match='fixes Clean'):
            entry.main([flag, *OLD])


def test_task_order_participates_in_identity():
    cfg = ExperimentConfig(method='dasr', task_sequence=OLD)
    old = training.digest(training.protocol_settings(cfg, True, 1., False))
    cfg.task_sequence = NEW
    assert old != training.digest(training.protocol_settings(cfg, True, 1., False))


def tiny_config(root, mode):
    torch.set_num_threads(2)
    return ExperimentConfig(
        method='dasr_uniform_' + mode, dataset='cifar10', output_root=str(root),
        run_name='sequence_test', device='cpu', workers=0, batch_size=4,
        depth=10, widen_factor=1, dropout=0., task_sequence=NEW,
        # Exercise several eligible epochs after the shared warmup.
        task_epochs=dict.fromkeys(NEW, 8), train_attack_steps=1, eval_attack_steps=1,
        cw_eval_steps=1, max_train_batches=1, max_eval_batches=1,
        final_test_methods=NEW,
    )


def execute(cfg, mode, policy='uniform', **kwargs):
    return training.run_experiment(cfg, policy=policy, smoke=True,
        replay_source_mode=mode, far_monitor_size=4, far_history_size=4, **kwargs)


@pytest.fixture(scope='module', params=SOURCE_MODES)
def completed(request, tmp_path_factory):
    mode = request.param
    cfg = tiny_config(tmp_path_factory.mktemp('sequence_' + mode), mode)
    execute(cfg, mode)
    return cfg, mode


def test_real_training_seen_tasks_replay_and_selected_models(completed):
    cfg, mode = completed
    rows = json.loads((cfg.run_dir / 'epoch_metrics.json').read_text())
    assert [row['task'] for row in rows] == [t for t in NEW for _ in range(8)]
    for row in rows:
        stage = row['stage']
        assert set(row['validation']) == set(NEW[:stage + 1])
        assert set(row['fg_monitor']) == set(NEW[:stage])
        assert row['replay_target'] is None or row['replay_target'] in NEW[:stage]
        if stage and row['epoch']:
            assert row['replay_target'] in NEW[:stage]
    for stage, task in enumerate(NEW):
        selected = torch.load(cfg.run_dir / f'stage{stage}_{task}_selected.pt',
                              map_location='cpu', weights_only=False)
        assert selected['settings']['config']['task_sequence'] == NEW
        from dasr_aligned.selection import best_eligible
        assert best_eligible(cfg, task, selected['epoch'])
    bank = torch.load(cfg.run_dir / 'far_history_bank.pt', map_location='cpu', weights_only=False)
    assert set(bank['tasks']) == set(NEW)
    for stage, task in enumerate(NEW):
        assert bank['tasks'][task]['checkpoint'] == f'stage{stage}_{task}_selected.pt'
    if mode == 'regenerated':
        assert all('training_adv' not in record for record in bank['tasks'].values())
        assert sum(row['historical_replay_examples'] for row in rows) == 0
    if mode == 'historical':
        assert sum(row['regenerated_replay_examples'] for row in rows) == 0
        assert sum(row['historical_replay_examples'] for row in rows) > 0
    assert set(json.loads((cfg.run_dir / 'final_test.json').read_text())) == set(NEW)


def test_old_sequence_cannot_resume_new_run(completed):
    cfg, mode = completed
    other = copy.deepcopy(cfg)
    other.task_sequence = OLD
    other.resume = True
    manifest = (cfg.run_dir / 'comparison_config.json').read_bytes()
    with pytest.raises(ValueError, match='mismatch|configuration|task|stage'):
        execute(other, mode, resume_mode='epoch')
    assert (cfg.run_dir / 'comparison_config.json').read_bytes() == manifest


def test_epoch_resume_with_new_sequence_is_reproducible(tmp_path, monkeypatch):
    mode = 'fg_adaptive_lazy'
    baseline = tiny_config(tmp_path / 'baseline', mode)
    execute(baseline, mode)
    resumed = tiny_config(tmp_path / 'interrupted', mode)
    save = training.atomic_checkpoint

    def interrupt(path, payload):
        save(path, payload)
        if Path(path).name == 'stage2_pgd_l2_resume.pt' and payload['epoch'] == 6:
            raise KeyboardInterrupt('test interruption after complete checkpoint write')

    monkeypatch.setattr(training, 'atomic_checkpoint', interrupt)
    with pytest.raises(KeyboardInterrupt):
        execute(resumed, mode)
    monkeypatch.setattr(training, 'atomic_checkpoint', save)
    resumed.resume = True
    execute(resumed, mode, resume_mode='epoch')
    for stage, task in enumerate(NEW):
        path = f'stage{stage}_{task}_selected.pt'
        left = torch.load(baseline.run_dir / path, map_location='cpu', weights_only=False)
        right = torch.load(resumed.run_dir / path, map_location='cpu', weights_only=False)
        assert left['epoch'] == right['epoch']
        assert all(torch.equal(value, right['model'][name]) for name, value in left['model'].items())
    assert json.loads((baseline.run_dir / 'final_test.json').read_text()) == json.loads(
        (resumed.run_dir / 'final_test.json').read_text())


def test_dtr_new_sequence_completes(tmp_path):
    cfg = tiny_config(tmp_path, 'fg_adaptive_lazy')
    execute(cfg, 'fg_adaptive_lazy', policy='dtr')
    rows = json.loads((cfg.run_dir / 'epoch_metrics.json').read_text())
    assert rows[-1]['task'] == 'patch'
    assert all(row['replay_target'] is None or row['replay_target'] in NEW[:row['stage']]
               for row in rows)


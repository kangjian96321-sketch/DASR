"""Integration regression: actual CPU training, interruption, and queue replay."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_replay_comparison as training
import far_replay
from pace_aligned.common import ExperimentConfig
from summarize_replay_study import aggregate_resources


def config(output):
    cfg = ExperimentConfig(method='pace_uniform', dataset='cifar10',
        output_root=str(output), run_name='resume_integration', device='cpu',
        depth=10, widen_factor=1, batch_size=4, workers=0,
        task_sequence=('clean', 'patch', 'pgd_l2'),
        task_epochs={'clean': 3, 'patch': 3, 'pgd_l2': 3},
        train_attack_steps=1, eval_attack_steps=1, max_train_batches=1,
        max_eval_batches=1, final_test_methods=('clean',))
    return cfg


def assert_same(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_training_resume_restores_queue_and_reproduces_next_epochs(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    baseline = config(tmp_path / 'uninterrupted')
    training.run_experiment(baseline, policy='uniform', smoke=True)
    resumed = config(tmp_path / 'interrupted')
    real_save = training.atomic_checkpoint

    def interrupt_after_save(path, payload):
        real_save(path, payload)
        if Path(path).name == 'stage1_patch_resume.pt' and payload['epoch'] == 1:
            raise KeyboardInterrupt('Simulated interruption after durable epoch checkpoint')

    monkeypatch.setattr(training, 'atomic_checkpoint', interrupt_after_save)
    with pytest.raises(KeyboardInterrupt):
        training.run_experiment(resumed, policy='uniform', smoke=True)
    saved = torch.load(resumed.run_dir / 'stage1_patch_resume.pt', weights_only=False)
    assert sum(saved['queue']['class_counts']) == 8  # 4 clean keys, once per epoch.
    assert saved['epoch_rows'][-1]['replay_examples'] == 4
    assert saved['schedule_rows'][-1]['target'] == 'clean'
    monkeypatch.setattr(training, 'atomic_checkpoint', real_save)
    resumed.resume = True
    training.run_experiment(resumed, policy='uniform', smoke=True, resume_mode='epoch')
    original_references = {}
    for stage, task in enumerate(baseline.task_sequence):
        left = torch.load(baseline.run_dir / f'stage{stage}_{task}_resume.pt', weights_only=False)
        right = torch.load(resumed.run_dir / f'stage{stage}_{task}_resume.pt', weights_only=False)
        for key in ('model', 'ema', 'optimizer', 'scheduler', 'queue', 'selector',
                    'running_sim', 'epoch_rows', 'schedule_rows', 'best_score',
                    'first_learned_reference', 'reference_sources'):
            assert_same(left[key], right[key])
        selected = torch.load(resumed.run_dir / f'stage{stage}_{task}_selected.pt', weights_only=False)
        original_references[task] = selected['first_learned_reference'][task]
        assert selected['first_learned_reference'] == original_references
        assert right['reference'] == {h: original_references[h] for h in baseline.task_sequence[:stage]}
        for h, src in selected['reference_sources'].items():
            assert src['stage'] == baseline.task_sequence.index(h)
            assert src['checkpoint'] == f'stage{src["stage"]}_{h}_selected.pt'
        for row in right['epoch_rows']:
            assert set(row['validation']) == set(baseline.task_sequence[:stage+1])
            assert row['selection_score'] == sum(row['validation'].values()) / len(row['validation'])
        if stage:
            assert sum(right['queue']['class_counts']) == 12
    assert json.loads((baseline.run_dir/'final_test.json').read_text()) == json.loads(
        (resumed.run_dir/'final_test.json').read_text())
    assert json.loads((resumed.run_dir/'replay_references.json').read_text())['accuracies'] == original_references


def test_resource_summary_retains_unavailable_and_uses_sample_sd():
    row = dict(policy='dtr', stage=1, task='patch', training_wall_time_seconds=1,
               gpu_training_time_seconds=None, peak_gpu_memory_allocated_bytes=100,
               replay_examples=4, clean_replay_examples=0, replay_batches=1, active_replay_epochs=1)
    other = {**row, 'training_wall_time_seconds': 3, 'peak_gpu_memory_allocated_bytes': 200}
    result = aggregate_resources([row, other])[0]
    assert result['training_wall_time_seconds_mean'] == 2
    assert result['training_wall_time_seconds_std'] == pytest.approx(2**0.5)
    assert result['gpu_training_time_seconds_mean'] is None
    assert result['peak_gpu_memory_allocated_bytes_mean'] == 150


def test_reference_resets_to_previous_stage_best(tmp_path, monkeypatch):
    """DTR and F/G share a reference fixed at the preceding task boundary."""
    torch.set_num_threads(2)
    cfg = config(tmp_path)
    cfg.method = 'pace_dtr'
    current = {'stage': 0}
    real_train = training.train_epoch

    def train_with_stage(model, teacher, ema, optimizer, criterion, queue, loader,
                         config, stage, *args):
        current['stage'] = stage
        return real_train(model, teacher, ema, optimizer, criterion, queue, loader,
                          config, stage, *args)

    def controlled_evaluation(model, loader, tasks, config, device):
        return {task: 90.0 - 10 * current['stage'] - 3 * cfg.task_sequence.index(task)
                for task in tasks}

    monkeypatch.setattr(training, 'train_epoch', train_with_stage)
    monkeypatch.setattr(training, 'evaluate_seen', controlled_evaluation)

    def controlled_reused_validation(self, model, loader, tasks):
        values = controlled_evaluation(model, loader, tasks, cfg, cfg.device)
        return values, dict(values)

    monkeypatch.setattr(
        training.FARHistoryBank,
        'evaluate_validation',
        controlled_reused_validation,
    )
    real_establish = training.FARHistoryBank.establish_task

    def controlled_reference(self, task, *args, **kwargs):
        real_establish(self, task, *args, **kwargs)
        value = 90.0 - 10 * current['stage'] - 3 * cfg.task_sequence.index(task)
        self.tasks[task]['baseline_accuracy'] = value
        self.save()
        return value

    monkeypatch.setattr(training.FARHistoryBank, 'establish_task', controlled_reference)

    def controlled_stage_reference(self, model, tasks):
        return {
            task: (
                90.0 - 10 * current['stage']
                - 3 * cfg.task_sequence.index(task)
            )
            for task in tasks
        }

    monkeypatch.setattr(
        training.FARHistoryBank,
        'fixed_historical_accuracies',
        controlled_stage_reference,
    )

    def controlled_far_evaluation(self, model, tasks, fresh_accuracies,
                                  tracker=None,
                                  component_threshold_pp=0.0,
                                  baseline_accuracies=None):
        result = {}
        for task in tasks:
            current_accuracy = (
                90.0 - 10 * current['stage']
                - 3 * cfg.task_sequence.index(task)
            )
            result[task] = far_replay.decompose(
                baseline_accuracies[task],
                current_accuracy,
                fresh_accuracies[task],
                component_threshold_pp,
            )
        return result

    monkeypatch.setattr(
        training.FARHistoryBank, 'evaluate', controlled_far_evaluation
    )
    training.run_experiment(cfg, policy='dtr', smoke=True)
    refs = json.loads((cfg.run_dir/'replay_references.json').read_text())
    assert refs['accuracies'] == {'clean': 90.0, 'patch': 77.0, 'pgd_l2': 64.0}
    assert refs['mode'] == 'initial_task_best_checkpoint'
    assert refs['dtr_reference_mode'] == 'previous_stage_best_checkpoint'
    schedule = json.loads((cfg.run_dir/'replay_schedule.json').read_text())['epochs']
    last = next(row for row in schedule if row['stage'] == 2 and row['epoch'] == 1)
    assert last['previous_epoch_accuracy']['clean'] == 70.0
    assert last['reference_accuracy']['clean'] == 80.0
    assert last['cumulative_reference_accuracy']['clean'] == 90.0
    assert last['reference_mode'] == 'previous_stage_best_checkpoint'
    assert last['fg_metrics_used']['clean']['a_ss'] == 80.0
    assert last['maximum_drop_pp'] == 10.0

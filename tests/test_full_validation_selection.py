"""Full-set model selection, with deliberately misleading monitor scores."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dasr_aligned import common
from dasr_aligned.selection import OLD_SEQUENCE, NEW_SEQUENCE, SELECTION_POLICY


@pytest.mark.parametrize('sequence', [OLD_SEQUENCE, NEW_SEQUENCE])
@pytest.mark.parametrize('reuse_current', [False, True])
def test_selection_counts_every_validation_example_once(sequence, reuse_current, monkeypatch):
    # Monitor accuracy is 100%, but full-set accuracy is 40% (two of five).
    x = torch.zeros(5, 3, 2, 2)
    y = torch.tensor([0, 0, 1, 1, 1])
    dataset = TensorDataset(x, y)
    val = DataLoader(dataset, batch_size=2)
    monitor = DataLoader(Subset(dataset, [0, 1]), batch_size=2)
    bundle = common.DataBundle(val, val, val, monitor, val, [], list(range(5)))
    cfg = common.ExperimentConfig(method='test', dataset='cifar10', device='cpu',
                                  batch_size=2, selection_probe_size=2)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 10))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
        model[1].bias[0] = 1
    examples = Counter()

    def record_attack(model, images, labels, task, cfg, training):
        assert training is False
        examples[task] += labels.numel()
        return images

    monkeypatch.setattr(common, 'generate_attack', record_attack)
    current = sequence[-1]
    result = common.evaluate_for_selection(
        model, bundle, current, sequence, cfg, torch.device('cpu'),
        current_accuracy=40. if reuse_current else None)
    assert result == dict.fromkeys(sequence, 40.)
    assert examples == Counter({task: 5 for task in sequence
                                if not reuse_current or task != current})


@pytest.mark.parametrize('method', ['sseat', 'crt'])
@pytest.mark.parametrize('sequence', [OLD_SEQUENCE, NEW_SEQUENCE])
def test_native_best_uses_full_scores_without_duplicate_evaluation(
        method, sequence, tmp_path, monkeypatch):
    # Exercise the native trainer, EMA selection, checkpoint writes and resume.
    # After warmup, epoch 2 wins on the monitor; epoch 3 wins on the full set.
    from test_sseat_crt_new_sequence import synthetic_setup
    _, native, args, configs = synthetic_setup(method, tmp_path, monkeypatch)
    original_config = native.config_from_args

    def configure(name, parsed):
        cfg = original_config(name, parsed)
        cfg.task_sequence = sequence
        cfg.task_epochs = dict.fromkeys(sequence, 3)
        cfg.max_eval_batches = None
        return cfg

    monkeypatch.setattr(native, 'config_from_args', configure)
    dataset = TensorDataset(torch.rand(6, 3, 32, 32), torch.arange(6))
    val = DataLoader(dataset, batch_size=2)
    monitor = DataLoader(Subset(dataset, [0, 1]), batch_size=2)
    bundle = common.DataBundle(val, val, val, monitor, val, [], list(range(6)))
    monkeypatch.setattr(native, 'build_dataloaders', lambda cfg: bundle)
    active = {}
    train_epoch = native.train_epoch

    def observed_epoch(*a, **kw):
        active['stage'] = a[7] if method == 'sseat' else kw['stage']
        active['epoch'] = a[8] if method == 'sseat' else kw['epoch']
        return train_epoch(*a, **kw)

    calls = Counter()

    def controlled_accuracy(model, loader, attack, cfg, device):
        stage, epoch = active['stage'], active['epoch']
        calls[stage, epoch, attack] += 1
        if loader is monitor:
            return [100., 100., 0.][epoch]
        assert loader is val
        return [100., 40., 80.][epoch]

    monkeypatch.setattr(native, 'train_epoch', observed_epoch)
    monkeypatch.setattr(common, 'evaluate_accuracy', controlled_accuracy)
    monkeypatch.setattr(native, 'final_evaluation', lambda *a, **k: {'clean': 80.})
    # Use the native main; the reversed wrapper intentionally locks its order.
    monkeypatch.setattr(sys, 'argv', ['train', *args])
    native.main()
    cfg = configs[-1]
    expected_calls = Counter({(stage, epoch, task): 1
                              for stage in range(4) for epoch in range(3)
                              for task in sequence[:stage + 1]})
    assert calls == expected_calls
    for stage, task in enumerate(sequence):
        payload = torch.load(cfg.run_dir / f'stage_{stage:02d}_{task}_best.pt',
                             map_location='cpu', weights_only=False)
        assert payload['epoch'] == 2  # zero-based: the third epoch
        assert payload['best_score'] == 80.
        assert payload['selection_policy'] == SELECTION_POLICY
    with (cfg.run_dir / 'metrics.csv').open(newline='', encoding='utf8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    for row in rows:
        assert row['val_mean'] == row['selection_mean']
        for task in sequence[:int(row['stage']) + 1]:
            assert row[f'val_{task}'] == row[f'selection_{task}']
    monkeypatch.setattr(native, 'train_epoch', lambda *a, **k: pytest.fail('Resume retrained'))
    monkeypatch.setattr(sys, 'argv', ['train', *args, '--resume'])
    native.main()
    assert calls == expected_calls


def test_old_selection_protocol_is_rejected_before_metadata_change(tmp_path):
    cfg = common.ExperimentConfig(method='sseat', run_directory=str(tmp_path / 'run'),
                                  resume=True)
    cfg.run_dir.mkdir()
    path = cfg.run_dir / 'config.json'
    saved = common.config_to_dict(cfg)
    saved['selection_policy'] = 'post_warmup_mean_all_seen_including_zero_v1_20261025'
    path.write_text(json.dumps(saved), encoding='utf8')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='selection policy mismatch'):
        common.prepare_run_directory(cfg)
    assert path.read_bytes() == before

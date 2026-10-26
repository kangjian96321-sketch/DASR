"""New sequence entries only. Synthetic CPU tests are not accuracy experiments."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dasr_aligned import common, crt, sseat
import train_crt_clean_linf_l2_patch as crt_entry
import train_sseat_clean_linf_l2_patch as sseat_entry
from sequence_clean_linf_l2_patch import TASK_SEQUENCE, validate_resume_sequence


OLD = ('clean', 'patch', 'pgd_l2', 'pgd')
ENTRIES = {'sseat': (sseat_entry, sseat), 'crt': (crt_entry, crt)}


@pytest.fixture(autouse=True)
def small_cpu_thread_count():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize('method', ENTRIES)
@pytest.mark.parametrize('dataset,backbone,batch', [
    ('imagenette', 'resnet18', 32), ('cifar100', 'wrn', 64), ('cifar10', 'wrn', 64),
])
def test_only_sequence_and_output_defaults_change(method, dataset, backbone, batch, monkeypatch):
    entry, native = ENTRIES[method]
    arguments = ['--dataset', dataset, '--backbone', backbone, '--batch-size', str(batch)]
    old = asdict(native.config_from_args(method, native.build_parser().parse_args(arguments)))
    captured = []
    monkeypatch.setattr(native, 'run', lambda cfg, *options: captured.append((cfg, options)))
    entry.main(arguments)
    cfg, options = captured[0]
    new = asdict(cfg)
    assert new['task_sequence'] == TASK_SEQUENCE and old['task_sequence'] == OLD
    assert cfg.output_root == './runs_replay_clean_linf_l2_patch'
    for key in ('task_sequence', 'output_root', 'run_name'):
        old.pop(key)
        new.pop(key)
    assert old == new
    assert cfg.task_epochs == {'clean': 100, 'patch': 30, 'pgd_l2': 30, 'pgd': 30}
    if method == 'crt':
        assert options == (crt.CRTOptions(),)


@pytest.mark.parametrize('method', ENTRIES)
def test_dry_run_never_trains_or_checks_resume(method, monkeypatch, capsys):
    entry, native = ENTRIES[method]
    monkeypatch.setattr(native, 'run', lambda *a: pytest.fail('Dry run started training'))
    entry.main(['--dry-run', '--resume', '--run-name', 'nonexistent_preview_only'])
    text = capsys.readouterr().out
    cfg, _ = json.JSONDecoder().raw_decode(text)
    assert cfg['task_sequence'] == list(TASK_SEQUENCE)


def test_original_entries_and_global_task_defaults_unchanged():
    import train_crt_dasr_aligned
    import train_sseat_dasr_aligned
    assert train_crt_dasr_aligned.main is crt.main
    assert train_sseat_dasr_aligned.main is sseat.main
    assert common.TASKS == OLD


def test_native_crt_options_forwarded(monkeypatch):
    captured = []
    monkeypatch.setattr(crt, 'run', lambda cfg, options: captured.append(options))
    crt_entry.main(['--crt-strategy', 'single', '--no-crt-use-alr', '--crt-alr-steps', '3',
                    '--crt-alr-weight-initial', '.75', '--crt-alr-weight-finetune', '.25'])
    assert captured == [crt.CRTOptions('single', False, 3, .75, .25)]


@pytest.mark.parametrize('args', [['--crt-alr-steps', '0'], ['--crt-alr-weight-initial', '-1']])
def test_native_crt_option_validation(args):
    with pytest.raises(SystemExit):
        crt_entry.main(args)


@pytest.mark.parametrize('method', ENTRIES)
@pytest.mark.parametrize('bad', ['old_sequence', 'missing_config', 'wrong_method'])
def test_resume_guard_before_run_or_metadata_writes(method, bad, tmp_path, monkeypatch):
    entry, native = ENTRIES[method]
    cfg = common.ExperimentConfig(method=method, output_root=str(tmp_path), run_name='guard',
                                 resume=True, task_sequence=TASK_SEQUENCE)
    cfg.run_dir.mkdir(parents=True)
    path = cfg.run_dir / 'config.json'
    saved = common.config_to_dict(cfg)
    if bad == 'old_sequence':
        saved['task_sequence'] = list(OLD)
    if bad == 'wrong_method':
        saved['method'] = 'sseat' if method == 'crt' else 'crt'
    if bad != 'missing_config':
        path.write_text(json.dumps(saved))
    before = path.read_bytes() if path.exists() else None
    monkeypatch.setattr(native, 'config_from_args', lambda *a: cfg)
    monkeypatch.setattr(native, 'run', lambda *a: pytest.fail('Guard ran after training runner'))
    with pytest.raises((ValueError, FileNotFoundError)):
        entry.main(['--resume'])
    assert (path.read_bytes() if path.exists() else None) == before


def synthetic_setup(method, root, monkeypatch):
    entry, native = ENTRIES[method]
    real_config = native.config_from_args
    captured_cfg = []

    def configure(name, args):
        cfg = real_config(name, args)
        cfg.depth, cfg.widen_factor, cfg.dropout = 10, 1, 0.
        cfg.task_epochs = dict.fromkeys(TASK_SEQUENCE, 1)
        cfg.train_attack_steps = cfg.eval_attack_steps = cfg.cw_eval_steps = 1
        cfg.max_train_batches = cfg.max_eval_batches = 1
        cfg.final_test_methods = TASK_SEQUENCE
        cfg.sseat_memory_size = 8
        cfg.sseat_replay_batch_size = 2
        cfg.sseat_uncertainty_views = 2
        captured_cfg.append(cfg)
        return cfg

    generator = torch.Generator().manual_seed(123)
    images = torch.rand(4, 3, 32, 32, generator=generator)
    labels = torch.arange(4)
    train = DataLoader(TensorDataset(images, labels, torch.arange(4)), batch_size=2)
    val = DataLoader(TensorDataset(images, labels), batch_size=2)
    bundle = common.DataBundle(train, val, val, val, val, list(range(4)), list(range(4, 8)))
    monkeypatch.setattr(native, 'config_from_args', configure)
    monkeypatch.setattr(native, 'build_dataloaders', lambda cfg: bundle)
    arguments = ['--dataset', 'cifar10', '--backbone', 'wrn', '--batch-size', '2',
                 '--device', 'cpu', '--workers', '0', '--output-root', str(root), '--run-name', 'synthetic']
    return entry, native, arguments, captured_cfg


@pytest.mark.parametrize('method', ENTRIES)
def test_real_four_stages_and_completed_stage_resume(method, tmp_path, monkeypatch):
    entry, native, args, configs = synthetic_setup(method, tmp_path, monkeypatch)
    original_epoch = native.train_epoch
    trained = []

    def observed_epoch(*a, **kw):
        task = a[6] if method == 'sseat' else kw['current_task']
        stage = a[7] if method == 'sseat' else kw['stage']
        if method == 'crt':
            assert kw['known_attacks'] == TASK_SEQUENCE[1:stage + 1]
        trained.append(task)
        return original_epoch(*a, **kw)

    monkeypatch.setattr(native, 'train_epoch', observed_epoch)
    result = entry.main(args)
    cfg = configs[-1]
    assert trained == list(TASK_SEQUENCE)
    summaries = json.loads((cfg.run_dir / 'stage_summaries.json').read_text())
    assert [row['task'] for row in summaries] == list(TASK_SEQUENCE)
    assert json.loads((cfg.run_dir / 'config.json').read_text())['task_sequence'] == list(TASK_SEQUENCE)
    for stage, task in enumerate(TASK_SEQUENCE):
        assert (cfg.run_dir / f'stage_{stage:02d}_{task}_best.pt').exists()
    if method == 'crt':
        assert [row['alr_weight'] for row in summaries] == [0., 1., .5, .5]
        assert summaries[-1]['known_attacks'] == ['pgd', 'pgd_l2', 'patch']
    else:
        memory = torch.load(cfg.run_dir / 'adr_memory.pt', map_location='cpu', weights_only=False)
        assert set(memory['task_ids'].tolist()).issubset({0, 1, 2, 3})
        assert memory['labels'].numel() > 0
    monkeypatch.setattr(native, 'train_epoch', lambda *a, **k: pytest.fail('Completed resume retrained'))
    assert entry.main([*args, '--resume']) == result


@pytest.mark.parametrize('method', ENTRIES)
def test_interrupted_stage_restarts_from_last_completed_boundary(method, tmp_path, monkeypatch):
    entry, native, args, configs = synthetic_setup(method, tmp_path, monkeypatch)
    original_epoch = native.train_epoch

    def interrupt(*a, **kw):
        task = a[6] if method == 'sseat' else kw['current_task']
        if task == 'pgd_l2':
            raise RuntimeError('synthetic interruption before stage2 first step')
        return original_epoch(*a, **kw)

    monkeypatch.setattr(native, 'train_epoch', interrupt)
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        entry.main(args)
    cfg = configs[-1]
    saved = torch.load(cfg.run_dir / 'completed_stage.pt', map_location='cpu', weights_only=False)
    assert saved['stage'] == 1
    after = []

    def resume_epoch(*a, **kw):
        after.append(a[6] if method == 'sseat' else kw['current_task'])
        return original_epoch(*a, **kw)

    monkeypatch.setattr(native, 'train_epoch', resume_epoch)
    entry.main([*args, '--resume'])
    assert after == ['pgd_l2', 'patch']
    assert len(json.loads((cfg.run_dir / 'stage_summaries.json').read_text())) == 4

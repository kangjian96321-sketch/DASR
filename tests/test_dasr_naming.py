"""Public names and import paths for the rename-only release."""
import importlib
import json
from pathlib import Path

import pytest

from ablation_config import LABELS, method_name
import train_DASR as training
import train_DASR_clean_linf_l2_patch as reverse


@pytest.mark.parametrize('entry,expected', [
    (training, ['clean', 'patch', 'pgd_l2', 'pgd']),
    (reverse, ['clean', 'pgd', 'pgd_l2', 'patch']),
])
def test_dasr_cli_name_and_sequence(entry, expected, capsys):
    entry.main(['--dry-run', '--dataset', 'imagenette', '--backbone', 'resnet18',
                '--batch-size', '32', '--autoattack-samples', '0',
                '--autoattack-batch-size', '4'])
    value = json.loads(capsys.readouterr().out)
    assert value['config']['method'] == 'dasr_dtr_fg_adaptive_lazy'
    assert value['task_sequence'] == expected
    assert value['settings']['far_replay']['monitor_size'] == 500
    assert value['settings']['far_replay']['history_size'] == 500
    assert value['settings']['version'].startswith('far_dasr_')
    assert value['config']['batch_size'] == 32


@pytest.mark.parametrize('module', [
    'dasr_components', 'dasr_aligned.common', 'dasr_aligned.crt',
    'dasr_aligned.sseat', 'dasr_aligned.joint', 'dasr_aligned.continual_baselines',
    'train_crt_dasr_aligned', 'train_sseat_dasr_aligned', 'train_joint_dasr_aligned',
])
def test_renamed_module_imports(module):
    assert importlib.import_module(module) is not None


def test_method_labels_and_removed_entry_files():
    assert LABELS['full'] == 'DASR'
    assert method_name('dtr') == 'dasr_dtr'
    assert method_name('uniform', 'no_queue') == 'dasr_uniform_no_queue'
    root = Path(__file__).resolve().parents[1]
    legacy = 'pa' + 'ce'
    assert not (root / (legacy + '_aligned')).exists()
    assert not (root / (legacy + '_components.py')).exists()
    assert not (root / ('train_' + legacy + '_clean_linf_l2_patch.py')).exists()
    assert not (root / ('train_' + 'replay_comparison.py')).exists()

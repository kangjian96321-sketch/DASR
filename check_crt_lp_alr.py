"""Compare our Lp ALR port with the actual pinned upstream L2Reg class.

Only the class is extracted, avoiding unrelated third-party attack imports.
Its hard-coded .cuda() calls are replaced by .to(selected_device); all math is
left intact. Use trusted source only. This checks one ALR call, not the whole
official training program. No dataset or CUDA is required for the CPU check.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from torch import nn
from pace_aligned.common import ExperimentConfig, isolated_random_seed
from pace_aligned.crt_lp_alr import _maximize, lp_parameters, official_lp_alr

REFERENCE_COMMIT = '5a6932430219d531aab5736de94292b149ff73ae'
REFERENCE_SHA256 = '87ba0a7f348ab802f03fd31156abc860a0809147bb9b1fc0dc1bbd56db9a937d'


def load_reference(repo, device, reference_file=None):
    """Read the pinned git blob, not a possibly edited working-tree file."""
    source = (Path(reference_file).read_bytes() if reference_file else
              subprocess.check_output(['git', '-C', str(Path(repo).resolve()),
                  'show', f'{REFERENCE_COMMIT}:train/reg_utils.py']))
    if hashlib.sha256(source).hexdigest() != REFERENCE_SHA256:
        raise ValueError('Reference source hash does not match the pinned upstream file')
    tree = ast.parse(source.decode('utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'L2Reg')

    class PortableDevice(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
                if node.args or node.keywords:
                    raise ValueError('Unexpected upstream cuda arguments')
                node.func.attr = 'to'
                node.args = [ast.Name(id='_reference_device', ctx=ast.Load())]
            return node

    module = ast.fix_missing_locations(PortableDevice().visit(
        ast.Module(body=[cls], type_ignores=[])))
    namespace = dict(torch=torch, nn=nn, _reference_device=device)
    exec(compile(module, '<pinned-upstream-L2Reg>', 'exec'), namespace)
    return namespace['L2Reg'], hashlib.sha256(source).hexdigest()


class ProbeModel(nn.Module):
    def __init__(self, stochastic=True):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 4, 1),
            nn.BatchNorm2d(4) if stochastic else nn.Identity(), nn.Tanh(),
            nn.Dropout(.3) if stochastic else nn.Identity(),
            nn.Flatten(), nn.Linear(4 * 4 * 4, 5))

    def forward(self, x):
        return self.layers(x)


class ZeroInputGradient(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(-.1, .1, 5))

    def forward(self, x):
        return self.bias[None, :] + 0 * x.flatten(1).sum(1, keepdim=True)


def compare_case(reference_class, task, steps, training, model_kind, device):
    torch.manual_seed(371)
    factory = ZeroInputGradient if model_kind == 'zero_gradient' else (
        lambda: ProbeModel(stochastic=model_kind == 'bn_dropout'))
    original = factory().to(device).train(training)
    cfg = ExperimentConfig(method='crt', l2_eps=.5, linf_eps=8 / 255)
    x = torch.rand(4, 3, 4, 4, device=device)
    x[0].zero_()  # Exercise the un-clamped initial perturbation at image bounds.
    x[1].fill_(1)
    epsilon, alpha = lp_parameters(task, cfg, steps)
    norm = 'L2Attack' if task == 'pgd_l2' else 'LinfAttack'
    seed = 137
    worst_difference = 0.0

    def compare(a, b):
        nonlocal worst_difference
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        worst_difference = max(worst_difference, float((a.float() - b.float()).abs().max()))

    port, upstream = copy.deepcopy(original), copy.deepcopy(original)
    ref = reference_class(upstream, epsilon, norm, steps, alpha, var_reg=False)
    with isolated_random_seed(seed, device), torch.no_grad():
        ref_view, _ = ref.get_max(x)
    with isolated_random_seed(seed, device):
        our_view = _maximize(port, x, task, cfg, steps)
    compare(our_view, ref_view)
    for key in upstream.state_dict():
        compare(port.state_dict()[key], upstream.state_dict()[key])

    port, upstream = copy.deepcopy(original), copy.deepcopy(original)
    ref = reference_class(upstream, epsilon, norm, steps, alpha, var_reg=False)
    with isolated_random_seed(seed, device):
        ref_loss = ref(x)
    # Upstream training discards inner weight gradients before outer backward.
    upstream.zero_grad(set_to_none=True)
    our_loss = official_lp_alr(port, x, task, cfg, steps, seed)
    assert all(p.grad is None for p in port.parameters())
    compare(our_loss, ref_loss)
    ref_loss.backward()
    our_loss.backward()
    for p, q in zip(port.parameters(), upstream.parameters()):
        compare(p.grad, q.grad)
    for key in upstream.state_dict():
        compare(port.state_dict()[key], upstream.state_dict()[key])
    assert port.training == upstream.training == training
    return dict(task=task, steps=steps, training=training, model=model_kind,
                maximum_absolute_difference=worst_difference, passed=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--crt-repo')
    source_group.add_argument('--reference-file', help='Exact SHA256-checked upstream reg_utils.py')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--report', help='Optional JSON result path')
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    reference, source_hash = load_reference(args.crt_repo, device, args.reference_file)
    cases = [compare_case(reference, task, steps, mode, kind, device)
             for task in ('pgd_l2', 'pgd') for steps in (1, 2, 3, 4, 10)
             for mode in (False, True)
             for kind in ('smooth', 'bn_dropout', 'zero_gradient')]
    report = dict(reference_commit=REFERENCE_COMMIT, source_sha256=source_hash,
        source='train/reg_utils.py:L2Reg(var_reg=False)', device=str(device),
        torch_version=torch.__version__, dtype='float32',
        reference_adjustment='only .cuda() -> .to(selected_device)',
        scope='single ALR call: perturbation, loss, outer gradients and BN state',
        passed=len(cases), cases=cases)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, indent=2))
    print('Maximum absolute difference:', max(c['maximum_absolute_difference'] for c in cases))


if __name__ == '__main__':
    main()

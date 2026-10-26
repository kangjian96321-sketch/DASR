"""Inspect trusted local checkpoint files, validate weights, and plan continuation.

Loading PyTorch pickle checkpoints requires trusting the file's producer.
This module never changes an imported source checkpoint.
"""
from __future__ import annotations
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import torch

TASKS = ('clean', 'patch', 'pgd_l2', 'pgd')
ALIASES = {'pgd_linf': 'pgd', 'pgd-linf': 'pgd', 'pgd_inf': 'pgd',
           'pgd-l2': 'pgd_l2', 'pgdl2': 'pgd_l2'}

def task_index(value, tasks=TASKS):
    if value is None:
        return None
    text = str(value).lower()
    task = ALIASES.get(text, text)
    if task.isdigit():
        index = int(task)
        if 0 <= index < len(tasks):
            return index
    if task in tasks:
        return tasks.index(task)
    raise ValueError(f'Unknown task {value!r}; choose 0..{len(tasks)-1} or {list(tasks)}')

def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()

def read_checkpoint(path):
    value = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f'Expected a checkpoint dictionary: {path}')
    return value

def json_read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def clean_keys(state):
    result = {}
    for name, tensor in state.items():
        key = name[7:] if name.startswith('module.') else name
        if key in result:
            raise ValueError(f'Duplicate parameter after removing module. prefix: {key}')
        result[key] = tensor
    return result

def validated_state(payload, model, use_ema=False):
    raw = payload.get('model', payload.get('model_state_dict', payload.get('state_dict')))
    if raw is None and payload and all(torch.is_tensor(v) for v in payload.values()):
        raw = payload
    if not isinstance(raw, dict):
        raise ValueError('Missing model/model_state_dict/state_dict or raw tensor state_dict')
    state = clean_keys(raw)
    expected = model.state_dict()
    missing, extra = set(expected)-set(state), set(state)-set(expected)
    wrong = [k for k in expected.keys() & state.keys()
             if not torch.is_tensor(state[k]) or tuple(state[k].shape) != tuple(expected[k].shape)]
    if missing or extra or wrong:
        raise ValueError(f'Checkpoint architecture mismatch: missing={sorted(missing)[:8]}, '
                         f'extra={sorted(extra)[:8]}, wrong_shape={wrong[:8]}. '
                         'Check --backbone, --dataset, and WRN --depth/--widen-factor; '
                         'WRN and ResNet checkpoints cannot be interchanged.')
    for key in ('mean', 'std'):
        if key in expected and not torch.allclose(state[key].float().cpu(), expected[key].float().cpu()):
            raise ValueError(f'Checkpoint normalization buffer {key} differs from --dataset')
    if use_ema:
        ema = payload.get('ema', payload.get('ema_state_dict'))
        if ema is not None:
            ema = clean_keys(ema)
            parameters = dict(model.named_parameters())
            if not set(parameters) <= set(ema) or not set(ema) <= set(expected):
                raise ValueError('EMA must contain all trainable parameters with compatible names')
            for name in parameters:
                if tuple(ema[name].shape) != tuple(expected[name].shape):
                    raise ValueError(f'EMA shape mismatch: {name}')
                state[name] = ema[name]
    return {key: value.detach().cpu().clone() for key, value in state.items()}

@dataclass
class Checkpoint:
    path: Path
    stage: int
    task: str
    kind: str
    epoch: int
    completed: bool
    native: bool
    selected_model: bool
    identity: dict | None
    sha256: str

    def info(self):
        return {**self.__dict__, 'path': str(self.path.resolve())}

def boundary_payload(payload, ref):
    """Use the selected EMA snapshot, rather than last weights, at a boundary."""
    if ref.kind == 'resume' and payload.get('best_snapshot'):
        result = {**payload, **payload['best_snapshot']}
        result.pop('ema', None)
        result.pop('ema_state_dict', None)
        return result
    return payload


def inspect(path, tasks=TASKS, checkpoint_task=None):
    path = Path(path)
    current = re.fullmatch(r'stage(\d+)_(.+)_(selected|resume)\.(pt|pth|ckpt)', path.name, re.I)
    legacy = re.fullmatch(r'(best|last|resume)_Task(\d+)_(.+)\.(pt|pth|ckpt)', path.name, re.I)
    if not current and not legacy and checkpoint_task is None:
        return None
    value = read_checkpoint(path)
    if current:
        stage, label, kind = int(current[1]), current[2].lower(), current[3].lower()
    elif legacy:
        kind, stage, label = legacy[1].lower(), int(legacy[2]), legacy[3].lower()
    else:
        stage = task_index(checkpoint_task, tasks); label = tasks[stage]
        kind = 'resume' if ('scheduler_state_dict' in value or 'scheduler' in value) else 'best'
    if stage >= len(tasks) or ALIASES.get(label, label) != tasks[stage]:
        raise ValueError(f'Checkpoint task order differs from requested sequence: {path.name}')
    if 'stage' in value and int(value['stage']) != stage:
        raise ValueError(f'Checkpoint stage metadata differs from filename: {path}')
    if 'task' in value and value['task'] != tasks[stage]:
        raise ValueError(f'Checkpoint task metadata differs from filename: {path}')
    native = value.get('format_version') == 3
    completed = kind == 'last' or bool(value.get('completed', False))
    # v2 records this field only after the stage has actually finished.
    if kind == 'selected' and tasks[stage] in value.get('first_learned_reference', {}):
        completed = True
    done = path.parent / f'stage{stage}_{tasks[stage]}_complete.json'
    if current and kind == 'selected' and done.exists():
        meta = json_read(done)
        if (meta.get('identity') in (None, value.get('identity')) and
                (not meta.get('checkpoint_sha256') or meta['checkpoint_sha256'] == file_hash(path))):
            completed = True
    if kind == 'resume' and 'planned_epochs' in value:
        completed = int(value.get('epoch', -1)) + 1 >= int(value['planned_epochs'])
    return Checkpoint(path, stage, tasks[stage], kind, int(value.get('epoch', -1)),
                      completed, native, value.get('boundary_kind') != 'last' and
                      (kind not in ('last','resume') or bool(value.get('best_snapshot'))),
                      value.get('identity'), file_hash(path))

def discover(directory, tasks=TASKS, explicit_file=None, checkpoint_task=None):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Checkpoint directory does not exist: {directory}')
    ignored = set(json_read(directory/'ignored_checkpoints.json', []))
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in ('.pt', '.pth', '.ckpt'))
    result = []
    for path in files:
        selected_file = explicit_file is not None and path.resolve() == Path(explicit_file).resolve()
        ref = inspect(path, tasks, checkpoint_task if selected_file else None)
        if ref is not None and ref.sha256 not in ignored:
            result.append(ref)
    return result

def preferred(records, stage, resume=False):
    options = [r for r in records if r.stage == stage and (r.kind == 'resume' if resume else r.kind != 'resume')]
    if not options and not resume:
        options = [r for r in records if r.stage == stage and r.completed]
    if not options:
        return None
    def rank(r):
        canonical = r.path.name == f'stage{r.stage}_{r.task}_{r.kind}.pt'
        return (int(r.selected_model), int(r.native),
                {'selected': 3, 'best': 2, 'last': 1, 'resume': 0}[r.kind], int(canonical))
    options.sort(key=rank, reverse=True)
    top = [r for r in options if rank(r) == rank(options[0])]
    if len({r.sha256 for r in top}) > 1:
        raise ValueError(f'Multiple checkpoints for stage {stage}: {[str(r.path) for r in top]}. Use a single experiment directory.')
    return options[0]

def plan(records, tasks=TASKS, start_task=None, completed_through=None, evaluate_only=False):
    declared = task_index(completed_through, tasks)
    completed = {r.stage for r in records if r.completed}
    if declared is not None:
        # Declaring a boundary does not invent missing historical checkpoints.
        if preferred(records, declared) is None:
            raise ValueError('--completed-through requires a model at that task boundary')
        completed.update(range(declared+1))
    if evaluate_only:
        if preferred(records, len(tasks)-1) is None:
            raise ValueError('--eval-only requires a final-task checkpoint')
        return len(tasks), completed
    if start_task is not None:
        return task_index(start_task, tasks), completed
    if not records:
        return 0, completed
    latest = max(r.stage for r in records)
    if latest in completed:
        return latest+1, completed
    if preferred(records, latest, resume=True) is not None:
        return latest, completed
    raise ValueError(f'{tasks[latest]} has only a best/selected weight file without completion evidence. '
                     f'If training finished, use --completed-through {tasks[latest]}; '
                     f'to restart a task, use --start-task TASK.')

def archive_branch(run_dir, records, start_stage, tasks=TASKS, epoch_resume=False):
    """Archive generated outputs; retain and ignore superseded imported sources."""
    run_dir = Path(run_dir).resolve()
    boundary = start_stage + int(epoch_resume)
    names = [f'stage{i}_{task}_{suffix}' for i, task in enumerate(tasks) if i >= boundary
             for suffix in ('selected.pt', 'resume.pt', 'complete.json')]
    names += ['run_complete.json', 'final_test.json', 'final_test_meta.json',
              'accuracy_matrix.json', 'epoch_metrics.json', 'replay_schedule.json',
              'replay_references.json', 'resource_metrics.json', 'resource_metrics.csv']
    names += ['epoch_metrics.csv', 'accuracy_matrix.csv', 'final_test.csv', 'RESULTS_CN.md',
              'training_curves.png', 'validation_matrix.png', 'final_accuracy.png']
    existing = [run_dir/name for name in names if (run_dir/name).exists()]
    if existing:
        destination = run_dir/'backups'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8])
        destination.mkdir(parents=True)
        for path in existing:
            if path.parent.resolve() != run_dir:
                raise ValueError('Archive target is outside run directory')
            path.replace(destination/path.name)
    ignored = set(json_read(run_dir/'ignored_checkpoints.json', []))
    ignored.update(r.sha256 for r in records if r.stage >= boundary)
    write_json(run_dir/'ignored_checkpoints.json', sorted(ignored))

def missing_resource(stage, task, cfg, reason):
    return {'method': cfg.method, 'dataset': cfg.dataset, 'train_seed': cfg.seed,
            'stage': stage, 'task': task, 'training_wall_time_seconds': None,
            'gpu_training_time_seconds': None, 'peak_gpu_memory_allocated_bytes': None,
            'peak_gpu_memory_reserved_bytes': None, 'replay_storage_bytes': None,
            'replay_examples': None, 'clean_replay_examples': None, 'replay_batches': None,
            'active_replay_epochs': None, 'provenance': reason}

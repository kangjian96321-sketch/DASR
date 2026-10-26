"""PACE replay-task and replay-source comparison.

The historical task selector and the historical-vs-regenerated source router
are independent command-line choices. F/G histories use selected checkpoints.
No training starts on import. --smoke uses synthetic data and is NOT an experiment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from my_functions.backbone_options import add_backbone_argument, backbone_spec

from pace_components import (AdaptiveDecoupledLACDLoss, DecoupledLACDLoss,
                             TeacherFeatureQueue, compute_grad_cosine_sim)
from replay_policies import POLICIES, ReplaySelector, historical_tasks
from far_replay import (
    SOURCE_MODES, FARHistoryBank, choose_source, eager_training_cache,
    register_tracker, source_probabilities,
)
from ablation_config import ABLATIONS, AblationConfig, make_criterion, method_name
import checkpoint_compat as compat
from pace_aligned.selection import best_eligible, selection_mean, warmup_epochs
from pace_aligned.common import (
    AUTOATTACK_SAMPLING_POLICY,
    DataBundle, EMA, ExperimentConfig, StageResourceTracker, build_dataloaders,
    clone_frozen, evaluate_seen, final_evaluation, generate_attack,
    make_backbone, make_optimizer, make_scheduler, resolve_device, set_seed,
    tensor_storage_bytes, training_attack_seed, upsert_resource_row,
    write_environment_manifest, prepare_final_evaluation,
)

PROTOCOL_VERSION = 'far_pace_replay_sources_v2_balanced_reused_validation'
REFERENCE_MODE = 'previous_stage_best_checkpoint'
BASE_WEIGHTS = {'clean': 0.0, 'patch': 1.0, 'pgd_l2': 1.0, 'pgd': 3.0}
ADAPTIVE_TASKS = {'patch', 'pgd_l2', 'pgd'}
QUEUE_CAPACITY = 256


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def model_digest(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        result.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
        result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def epoch_loader(dataset, cfg, stage, epoch):
    # Fresh workers make interrupted and uninterrupted epoch boundaries agree.
    # Identical stages/epochs receive the same shuffle/augmentation seed in all arms.
    epoch_seed = cfg.seed * 1_000_000 + stage * 10_000 + epoch
    set_seed(epoch_seed)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                      num_workers=cfg.workers, pin_memory=cfg.device.startswith('cuda'),
                      persistent_workers=False, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(epoch_seed))


def protocol_settings(cfg, include_clean, drop_threshold, smoke, ablation=None,
                      replay_source_mode='fg_adaptive_lazy',
                      far_monitor_size=500, far_history_size=500,
                      far_component_threshold_pp=0.0, far_cache_dtype='float32'):
    ablation = ablation or AblationConfig()
    values = asdict(cfg)
    if cfg.backbone == 'wrn':
        # Legacy WRN protocols had no backbone key: retain their exact hashes.
        values.pop('backbone', None)
    else:
        for key in ('depth', 'widen_factor', 'dropout'):
            values.pop(key, None)  # WRN-only options are inactive for ResNet-18.
        values['resolved_backbone'] = backbone_spec(cfg)
    from pace_aligned.feature_plots import PLOT_FIELDS
    for key in PLOT_FIELDS:
        values.pop(key, None)
    for key in ('method', 'output_root', 'run_name', 'resume', 'device', 'data_root', 'run_directory', 'workers',
                'final_test_methods', 'autoattack_samples', 'autoattack_batch_size', 'autoattack_subset_seed'):
        values.pop(key, None)
    return {'version': PROTOCOL_VERSION, 'config': values,
            'include_clean': include_clean, 'drop_threshold_pp': drop_threshold,
            'ablation': ablation.to_dict(), 'queue_capacity': ablation.queue_capacity,
            'adaptive_tasks': sorted(ADAPTIVE_TASKS) if ablation.adaptive_temperature else [],
            'base_weights': BASE_WEIGHTS, 'selection_unit': 'epoch',
            'first_epoch_replay': False, 'reference_mode': REFERENCE_MODE, 'smoke': smoke,
            'far_replay': {
                'source_mode': replay_source_mode,
                'monitor_size': int(far_monitor_size),
                'history_size': int(far_history_size),
                'component_threshold_pp': float(far_component_threshold_pp),
                'cache_dtype': far_cache_dtype,
                'sampling': 'deterministic_class_balanced',
                'fresh_monitor_source': 'reused_full_validation',
                'temperature_normalization': 'per_view_group',
                'contrastive_negative_pool': 'joint',
            }}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_matched_schedule(path, expected_signature, expected_epochs):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('policy') != 'dtr' or payload.get('gate') != 'native':
        raise ValueError('Matched budgets must come from an independently run native DTR anchor')
    if not payload.get('complete') or not payload.get('history_complete', True) or payload.get('protocol_signature') != expected_signature:
        raise ValueError('DTR schedule is incomplete or its dataset/seed/protocol differs')
    entries = {}
    for row in payload['epochs']:
        key = (int(row['stage']), int(row['epoch']))
        if key in entries or type(row['active']) is not bool:
            raise ValueError('Duplicate/invalid DTR schedule entry')
        entries[key] = row['active']
    expected_keys = {(stage, epoch) for stage, count in enumerate(expected_epochs)
                     for epoch in range(count)}
    if set(entries) != expected_keys:
        raise ValueError('DTR schedule does not cover exactly the requested training epochs')
    return entries


def train_epoch(model, teacher, ema, optimizer, criterion, queue, loader,
                cfg, stage, epoch, task, decision, running_sim, tracker, ablation=None,
                history_bank=None, replay_source_mode='regenerated', far_metrics=None):
    ablation = ablation or AblationConfig()
    model.train()
    sums = {'loss': 0.0, 'classification': 0.0, 'acd': 0.0, 'weight': 0.0,
            'examples': 0, 'batches': 0, 'replay_examples': 0,
            'clean_replay_examples': 0, 'replay_batches': 0,
            'historical_replay_examples': 0, 'regenerated_replay_examples': 0,
            'historical_replay_batches': 0, 'regenerated_replay_batches': 0,
            'lazy_cache_builds': 0}
    with tracker.timed():
        for batch_index, batch in enumerate(loader):
            if cfg.max_train_batches is not None and batch_index >= cfg.max_train_batches:
                break
            x, y = batch[0].to(cfg.device), batch[1].to(cfg.device)
            batch_size = y.numel()
            current = generate_attack(model, x, y, task, cfg, training=True,
                                      rng_seed=training_attack_seed(cfg, stage, epoch, batch_index, 0))
            if task != 'clean':
                tracker.add_adversarial_samples('current_training', batch_size)
            views = [current]
            clean_views = [x]
            label_views = [y]
            if decision.active:
                source_seed = training_attack_seed(cfg, stage, epoch, batch_index, 2)
                source = choose_source(
                    replay_source_mode,
                    (far_metrics or {}).get(decision.target),
                    source_seed,
                )
                if source == 'historical':
                    if history_bank is None:
                        raise RuntimeError('Historical replay requires FARHistoryBank')
                    if history_bank.ensure_training_cache(
                        decision.target, tracker, active_model=model
                    ):
                        sums['lazy_cache_builds'] += 1
                    replay_clean, replay, replay_labels = history_bank.sample(
                        decision.target, batch_size, source_seed + 1
                    )
                else:
                    replay_clean, replay_labels = x, y
                    replay = generate_attack(
                        model, x, y, decision.target, cfg, training=True,
                        rng_seed=training_attack_seed(
                            cfg, stage, epoch, batch_index, 1
                        ),
                    )
                views.append(replay)
                clean_views.append(replay_clean)
                label_views.append(replay_labels)
                sums['replay_examples'] += batch_size
                sums['replay_batches'] += 1
                sums[source + '_replay_examples'] += batch_size
                sums[source + '_replay_batches'] += 1
                if decision.target == 'clean':
                    sums['clean_replay_examples'] += batch_size
                elif source == 'regenerated':
                    tracker.add_adversarial_samples('replay_training', batch_size)
            optimizer.zero_grad(set_to_none=True)
            logits, _, student_features, _ = model(torch.cat(views), return_feature=True)
            loss_cls = F.cross_entropy(logits[:batch_size], y)
            loss_acd = logits.new_zeros(())
            weight = 0.0
            if criterion is not None:
                with torch.no_grad():
                    key_model = model if ablation.key_source == 'student' else teacher
                    clean_keys = key_model(x, return_feature=True)[2].detach()
                    if len(clean_views) == 2 and clean_views[1] is not x:
                        replay_keys = key_model(
                            clean_views[1], return_feature=True
                        )[2].detach()
                        all_clean_keys = torch.cat([clean_keys, replay_keys])
                    else:
                        all_clean_keys = clean_keys.repeat(len(views), 1)
                combined_labels = torch.cat(label_views)
                criterion_kwargs = {
                    'queue_features': queue.features if queue is not None else None,
                    'queue_labels': queue.labels if queue is not None else None,
                }
                if isinstance(criterion, AdaptiveDecoupledLACDLoss):
                    criterion_kwargs['temperature_groups'] = tuple(
                        len(labels) for labels in label_views
                    )
                # Positives are paired with their own clean inputs. Temperature
                # min/max is group-specific, while negatives remain joint.
                loss_acd = criterion(
                    student_features, all_clean_keys, combined_labels,
                    **criterion_kwargs,
                )
                # One enqueue per original clean image; queue starts empty each task.
                if queue is not None:
                    queue.enqueue(clean_keys, y)
                if ablation.gradient_weighting and batch_index % 100 == 0:
                    similarity = compute_grad_cosine_sim(loss_cls, loss_acd, model)
                    running_sim = similarity if epoch == 0 and batch_index == 0 else (
                        0.8 * running_sim + 0.2 * similarity)
                weight = BASE_WEIGHTS[task] * (max(ablation.weight_floor, max(0.0, running_sim))
                                              if ablation.gradient_weighting else 1.0)
            loss = loss_cls + weight * loss_acd
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss: stage={stage}, epoch={epoch}, batch={batch_index}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update()
            for key, value in [('loss', loss.item()), ('classification', loss_cls.item()),
                               ('acd', loss_acd.item()), ('weight', weight)]:
                sums[key] += value * batch_size
            sums['examples'] += batch_size
            sums['batches'] += 1
    if not sums['examples']:
        raise RuntimeError('No training examples were processed')
    for key in ('loss', 'classification', 'acd', 'weight'):
        sums[key] /= sums['examples']
    if criterion is not None:
        criterion.epoch_distances.clear()
    return sums, running_sim


def synthetic_loaders(cfg):
    # Deterministic fixture used ONLY for executable smoke tests.
    generator = torch.Generator().manual_seed(9123)
    x = torch.rand(8, 3, 32, 32, generator=generator)
    y = torch.arange(8) % cfg.num_classes
    train = TensorDataset(x, y, torch.arange(8))
    evaluation = TensorDataset(x, y)
    tl = DataLoader(train, batch_size=cfg.batch_size)
    el = DataLoader(evaluation, batch_size=cfg.batch_size)
    return DataBundle(tl, el, el, el, el, list(range(8)), list(range(8)))


def assert_checkpoint_identity(ref, identity, settings, payload):
    """Native checkpoints require matching training settings; clean can be shared."""
    if not ref.native:
        return
    if ref.identity == identity:
        return
    stored = payload.get('settings')
    if ref.stage == 0 and stored:
        a, b = copy.deepcopy(stored), copy.deepcopy(settings)
        for item in (a, b):
            for key in ('ablation', 'adaptive_tasks', 'queue_capacity',
                        'drop_threshold_pp', 'include_clean', 'far_replay',
                        'reference_mode'):
                item.pop(key, None)
        if a == b:
            return
    raise ValueError(f'Checkpoint training configuration/policy mismatch: {ref.path}. '
                     'Use the original ablation and training settings in a separate run folder.')


def run_experiment(cfg, policy='dtr', include_clean=True, threshold=1.0,
                   match_schedule=None, smoke=False, ablation=None, resume_source=None,
                   start_task=None, resume_mode='task', completed_through=None,
                   checkpoint_task=None, eval_only=False, inspect_only=False,
                   replay_source_mode='fg_adaptive_lazy', far_monitor_size=500,
                   far_history_size=500, far_component_threshold_pp=0.0,
                   far_cache_dtype='float32'):
    ablation = ablation or AblationConfig()
    if resume_mode not in ('task', 'epoch'):
        raise ValueError('resume_mode must be task or epoch')
    if not ablation.replay and match_schedule:
        raise ValueError('only_cls/no_dtr cannot use --match-schedule')
    if policy not in POLICIES:
        raise ValueError(f'Unknown replay policy: {policy}')
    if replay_source_mode not in SOURCE_MODES:
        raise ValueError(f'Unknown replay source mode: {replay_source_mode}')
    if min(far_monitor_size, far_history_size) <= 0:
        raise ValueError('FAR monitor/history sizes must be positive')
    if not math.isfinite(far_component_threshold_pp) or far_component_threshold_pp < 0:
        raise ValueError('FAR component threshold must be finite and nonnegative')
    if far_monitor_size > cfg.val_size and not smoke:
        raise ValueError('FAR monitor size cannot exceed the validation-set size')
    # The balanced selection loader is also the prefix reused from full
    # validation, so programmatic callers must receive the same guarantee as
    # command-line callers.
    cfg.selection_probe_size = int(far_monitor_size)
    requested_resume = bool(cfg.resume or resume_source or start_task is not None or eval_only)
    effective_policy = policy if ablation.replay else 'none'
    settings = protocol_settings(
        cfg, include_clean, threshold, smoke, ablation,
        replay_source_mode, far_monitor_size, far_history_size,
        far_component_threshold_pp, far_cache_dtype,
    )
    signature = digest(settings)
    gate = 'matched' if match_schedule else 'native'
    identity = {'protocol_signature': signature, 'policy': policy, 'gate': gate,
                'match_schedule_sha256': compat.file_hash(match_schedule) if match_schedule else None}
    explicit_file = None
    source = Path(resume_source) if resume_source else cfg.run_dir
    if source.is_file():
        explicit_file, source = source, source.parent
    records = compat.discover(source, cfg.task_sequence, explicit_file, checkpoint_task) if requested_resume else []
    if requested_resume and source.resolve() != cfg.run_dir.resolve() and cfg.run_dir.exists():
        local_records = compat.discover(cfg.run_dir, cfg.task_sequence)
        local_stages = {r.stage for r in local_records if r.selected_model or r.kind == 'resume'}
        ignored = set(compat.json_read(cfg.run_dir/'ignored_checkpoints.json', []))
        records = local_records + [r for r in records if r.stage not in local_stages and r.sha256 not in ignored]
    begin, completed = compat.plan(records, cfg.task_sequence, start_task, completed_through, eval_only)
    if requested_resume and not records:
        if not (start_task is not None and compat.task_index(start_task, cfg.task_sequence) == 0
                and not cfg.resume and not resume_source and not eval_only):
            raise FileNotFoundError('No recognized checkpoint files. Check --resume PATH; '
                                    'for an arbitrary filename pass --checkpoint-task TASK.')
    if not requested_resume:
        begin = 0
    print(json.dumps({'start_task': cfg.task_sequence[begin] if begin < len(cfg.task_sequence) else 'final_test',
                      'resume_mode': resume_mode, 'ablation': ablation.to_dict(),
                      'checkpoints': [r.info() for r in records]}, indent=2), flush=True)
    if inspect_only:
        return cfg.run_dir
    device = resolve_device(cfg.device); cfg.device = str(device)
    set_seed(cfg.seed)
    model = make_backbone(cfg).to(device)
    checkpoint_imported_history = False
    # Validate every imported file before writing run metadata or archiving outputs.
    for ref in records:
        payload = compat.read_checkpoint(ref.path)
        validated = compat.validated_state(payload, model, use_ema=False)
        assert_checkpoint_identity(ref, identity, settings, payload)
        checkpoint_imported_history |= bool(payload.get('import_origin') or payload.get('imported_history'))
        del payload, validated
    manifest_path = cfg.run_dir/'comparison_config.json'
    previous = compat.json_read(manifest_path)
    if previous and previous.get('identity') != identity:
        raise ValueError('Output folder has another configuration; choose --run-name or --run-dir for a new run.')
    if cfg.run_dir.exists() and any(cfg.run_dir.iterdir()) and not requested_resume:
        raise FileExistsError('Run folder is not empty; use --resume or a new run name.')
    matched = (load_matched_schedule(match_schedule, signature,
        [cfg.task_epochs[t] for t in cfg.task_sequence]) if match_schedule else None)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    loaders = synthetic_loaders(cfg) if smoke else build_dataloaders(cfg)
    history_bank = (
        FARHistoryBank(
            cfg.run_dir/'far_history_bank.pt', cfg, loaders, device,
            monitor_size=far_monitor_size, training_size=far_history_size,
            cache_dtype=far_cache_dtype, smoke=smoke,
        )
        if ablation.replay else None
    )
    split_info = {'train': loaders.train_indices, 'validation': loaders.val_indices}
    split_signature = digest(split_info)
    old_split = compat.json_read(source/'data_split.json') if requested_resume else None
    if old_split and old_split['signature'] != split_signature:
        raise ValueError('Imported checkpoint data split differs from the requested split.')
    if match_schedule:
        anchor_split = compat.json_read(Path(match_schedule).parent/'data_split.json')
        if not anchor_split or anchor_split['signature'] != split_signature:
            raise ValueError('Dataset split differs from the matched DTR anchor')
    atomic_json(manifest_path, {'identity': identity, 'settings': settings,
        'data_root': str(Path(cfg.data_root).resolve()), 'status': 'smoke_test' if smoke else 'experiment'})
    write_environment_manifest(cfg)
    atomic_json(cfg.run_dir/'data_split.json', {'signature': split_signature, **split_info})
    import_records = compat.json_read(cfg.run_dir/'checkpoint_imports.json', [])
    for ref in records:
        info = ref.info()
        if not any(r['sha256'] == info['sha256'] for r in import_records):
            import_records.append({**info, 'declared_ablation': ablation.name,
                'source_config_known': ref.native,
                'completion_declared_through': completed_through})
    atomic_json(cfg.run_dir/'checkpoint_imports.json', import_records)
    sources_legacy = any(not r.native for r in records)
    legacy_origin = sources_legacy or checkpoint_imported_history or bool(previous and previous.get('imported_history'))
    # Load the resume payload BEFORE archiving anything. File-backed best snapshots
    # are embedded in future resume files so deleting sidecar JSON is recoverable.
    resume_ref = compat.preferred(records, begin, resume=True) if begin < len(cfg.task_sequence) else None
    saved = compat.read_checkpoint(resume_ref.path) if resume_mode == 'epoch' and resume_ref else None
    if resume_mode == 'epoch' and begin < len(cfg.task_sequence) and saved is None and any(r.stage == begin for r in records):
        raise ValueError('Epoch continuation needs a resume checkpoint; use --resume-mode task to restart.')
    exact_resume = bool(saved and resume_ref.native)
    first_learned_reference, reference_sources = {}, {}
    schedule_rows, stage_rows, accuracy_matrix, resource_rows = [], [], [], []
    previous_state = None
    history_complete = True
    # Repair completed boundary artifacts using the actual selected checkpoints.
    for stage, task in enumerate(cfg.task_sequence[:begin]):
        ref = compat.preferred(records, stage)
        if ref is None:
            history_complete = False
            accuracy_matrix.append({'stage': stage, 'task': task, 'selected_epoch': None,
                'accuracies': None, 'split': 'full_validation', 'provenance': 'checkpoint_missing'})
            resource_rows.append(compat.missing_resource(stage, task, cfg, 'checkpoint_missing'))
            continue
        payload = compat.boundary_payload(compat.read_checkpoint(ref.path), ref)
        selected_epoch = int(payload.get('epoch', ref.epoch))
        state = compat.validated_state(payload, model, use_ema=True)
        model.load_state_dict(state)
        done = payload.get('stage_summary') or compat.json_read(ref.path.parent/f'stage{stage}_{task}_complete.json')
        if done is None and ref.native and ref.kind == 'resume' and ref.completed:
            done = {'schedule_rows': payload.get('schedule_rows', []),
                    'epoch_rows': payload.get('epoch_rows', []),
                    'history_complete': payload.get('history_complete', False)}
        if done and done.get('identity') not in (None, identity):
            done = None
        trusted_validation = ref.native and payload.get('settings') == settings
        validation = payload.get('full_validation') if trusted_validation else None
        if validation is None:
            validation = evaluate_seen(model, loaders.val, cfg.task_sequence[:stage+1], cfg, device)
        known_reference = payload.get('first_learned_reference', {}) if ref.native else {}
        boundary_kind = payload.get('boundary_kind', 'last' if ref.kind == 'last' or
            (ref.kind == 'resume' and not payload.get('best_snapshot')) else 'selected')
        far_reference = None
        if history_bank is not None and boundary_kind != 'last':
            far_reference = history_bank.establish_task(
                task, model, f'stage{stage}_{task}_selected.pt',
                build_training=(
                    eager_training_cache(replay_source_mode)
                    and stage < len(cfg.task_sequence) - 1
                ),
            )
        if far_reference is not None:
            reference = far_reference
        elif task in known_reference:
            reference = known_reference[task]
        elif boundary_kind == 'last':
            reference = None
        else:
            reference = evaluate_seen(model, loaders.selection_val, [task], cfg, device)[task]
        if reference is not None:
            first_learned_reference[task] = reference
        elif begin < len(cfg.task_sequence) and effective_policy != 'none' and (task != 'clean' or include_clean):
            raise ValueError(f'First-learning references need the original best/selected checkpoint for {task}; '
                             'a last checkpoint only provides final-epoch weights.')
        source_info = {'stage': stage, 'selected_epoch': selected_epoch,
            'checkpoint': f'stage{stage}_{task}_selected.pt', 'source_sha256': ref.sha256,
            'split': 'selection_val', 'origin': 'native' if ref.native else 'imported_boundary_re_evaluated'}
        if reference is not None:
            reference_sources[task] = payload.get('reference_sources', {}).get(task, source_info) if ref.native else source_info
        matrix_row = {'stage': stage, 'task': task, 'selected_epoch': selected_epoch,
            'accuracies': validation, 'split': 'full_validation',
            'provenance': 'native' if trusted_validation else 're_evaluated_' + boundary_kind}
        accuracy_matrix.append(matrix_row)
        historical_schedule = done.get('schedule_rows', []) if done else []
        historical_epochs = done.get('epoch_rows', []) if done else []
        stage_history_complete = bool(done and done.get('history_complete', True))
        history_complete &= stage_history_complete
        schedule_rows.extend(historical_schedule); stage_rows.extend(historical_epochs)
        resource = done.get('resource_row') if done else None
        if resource is None:
            old_rows = compat.json_read(ref.path.parent/'resource_metrics.json', [])
            resource = next((r for r in old_rows if r.get('stage') == stage), None)
        resource = resource or compat.missing_resource(stage, task, cfg, 'historical_measurements_unavailable')
        resource_rows.append(resource)
        summary = {'identity': identity, 'schedule_rows': historical_schedule, 'epoch_rows': historical_epochs,
            'accuracy_matrix_row': matrix_row, 'resource_row': resource,
            'history_complete': stage_history_complete, 'completed': True}
        converted = {'format_version': 3, 'identity': identity, 'settings': settings,
            'stage': stage, 'task': task, 'completed': True, 'model': state, 'epoch': selected_epoch,
            'boundary_kind': boundary_kind,
            'imported_history': legacy_origin,
            'planned_epochs': cfg.task_epochs[task], 'score': sum(validation.values())/len(validation),
            'full_validation': validation, 'first_learned_reference': dict(first_learned_reference),
            'reference_sources': copy.deepcopy(reference_sources), 'stage_summary': summary,
            'import_origin': payload.get('import_origin', ref.info() if not ref.native else None)}
        destination = cfg.run_dir/f'stage{stage}_{task}_selected.pt'
        if not ref.native and ref.path.resolve() == destination.resolve():
            original = cfg.run_dir/'imported_originals'/ref.sha256/ref.path.name
            original.parent.mkdir(parents=True, exist_ok=True)
            if not original.exists():
                shutil.copy2(ref.path, original)
        atomic_checkpoint(destination, converted)
        atomic_json(cfg.run_dir/f'stage{stage}_{task}_complete.json', summary)
        previous_state = state if stage == begin-1 else previous_state
        del payload, state, converted
    # A self-contained native resume also carries the immutable references and
    # preceding teacher needed when older boundary files were not copied.
    if saved and resume_ref.native:
        first_learned_reference = saved['first_learned_reference']
        reference_sources = saved['reference_sources']
        if previous_state is None:
            previous_state = saved.get('teacher')
        if history_bank is not None:
            far_state = saved.get('far_history_state')
            if far_state is None:
                raise ValueError(
                    'Native FAR epoch resume is missing the self-contained history '
                    'bank; restart this task with --resume-mode task'
                )
            history_bank.load_state_dict(far_state)
    if history_bank is not None:
        # The paired monitor baseline is authoritative for D = F + G.
        for task, entry in history_bank.tasks.items():
            first_learned_reference[task] = float(entry['baseline_accuracy'])
    final_only = begin >= len(cfg.task_sequence)
    if not final_only and begin > 0 and previous_state is None and not (saved and not ablation.use_teacher):
        raise ValueError(f'To restart {cfg.task_sequence[begin]}, provide the preceding task selected checkpoint. '
                         'Native --resume-mode epoch can instead use its embedded teacher.')
    needed_history = historical_tasks(cfg.task_sequence, begin, include_clean) if not final_only else []
    if not final_only and effective_policy != 'none':
        missing = set(needed_history)-set(first_learned_reference)
        if missing:
            raise ValueError(f'First-learning references need the original best/selected task checkpoints: {sorted(missing)}. '
                             'The final model cannot reconstruct earlier first-learning accuracies.')
    if not final_only:
        # Retain only the chosen branch; old imports remain untouched but ignored
        # by subsequent scans. Earlier completed boundaries are preserved.
        compat.archive_branch(cfg.run_dir, records, begin, cfg.task_sequence, epoch_resume=bool(saved))
        if begin == 0:
            set_seed(cfg.seed); model = make_backbone(cfg).to(device)
        elif previous_state is not None:
            model.load_state_dict(previous_state)
    atomic_json(manifest_path, {'identity': identity, 'settings': settings,
        'data_root': str(Path(cfg.data_root).resolve()), 'status': 'smoke_test' if smoke else 'experiment',
        'imported_history': legacy_origin})
    for row in resource_rows:
        upsert_resource_row(cfg.run_dir, row)
    for stage in range(begin, len(cfg.task_sequence)):
        task = cfg.task_sequence[stage]
        selected_path = cfg.run_dir/f'stage{stage}_{task}_selected.pt'
        resume_path = cfg.run_dir/f'stage{stage}_{task}_resume.pt'
        completed_path = cfg.run_dir/f'stage{stage}_{task}_complete.json'
        teacher = clone_frozen(model) if stage and ablation.use_teacher else None
        if saved and stage == begin and saved.get('teacher') is not None and teacher is not None:
            teacher.load_state_dict(saved['teacher'])
        ema = EMA(model, cfg.ema_decay)
        optimizer = make_optimizer(model, task, cfg); scheduler = make_scheduler(optimizer, task, cfg)
        history = historical_tasks(cfg.task_sequence, stage, include_clean)
        reference = {}
        observed = {}
        selector = ReplaySelector(effective_policy, cfg.seed*1000+stage, threshold)
        criterion = make_criterion(ablation, task)
        queue = (TeacherFeatureQueue(ablation.queue_capacity, cfg.num_classes, device)
                 if criterion is not None and ablation.use_queue else None)
        tracker = StageResourceTracker(cfg.method, cfg, model, stage, task, device)
        register_tracker(tracker)
        running_sim, best_score, start_epoch = 0.5, -math.inf, 0
        far_metrics = {}
        local_schedule, local_rows, best_snapshot = [], [], None
        probe_seconds = validation_seconds = serialization_seconds = 0.0
        stage_history_complete = True
        recovery_notes = []
        if history:
            if saved and stage == begin and exact_resume:
                # Keep the task-boundary reference fixed across epoch resume;
                # the active model in the resume file is already mid-task.
                saved_reference = saved.get('reference')
                if saved_reference is None:
                    raise ValueError(
                        'Native FAR epoch resume is missing its previous-stage '
                        'reference accuracies; restart this task with '
                        '--resume-mode task'
                    )
                missing_reference = set(history) - set(saved_reference)
                if missing_reference:
                    raise ValueError(
                        'Native FAR epoch resume is missing previous-stage '
                        f'references for: {sorted(missing_reference)}'
                    )
                reference = {
                    name: float(saved_reference[name]) for name in history
                }
            elif history_bank is not None:
                # Before the first optimizer step of a stage, model is the
                # selected/best checkpoint from the immediately preceding task.
                started = time.perf_counter()
                reference = history_bank.fixed_historical_accuracies(
                    model, history
                )
                probe_seconds += time.perf_counter() - started
            else:
                # Replay-disabled ablations do not use this mapping, but keep
                # their bookkeeping backward compatible.
                reference = {
                    name: first_learned_reference[name]
                    for name in history
                    if name in first_learned_reference
                }
        if saved and stage == begin:
            if exact_resume:
                model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
                optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
                if queue is not None:
                    if saved.get('queue') is None:
                        raise ValueError('Native checkpoint is missing its enabled queue state')
                    queue.load_state_dict(saved['queue'])
                selector.load_state_dict(saved['selector']); tracker.load_state_dict(saved['resources'])
                if criterion is not None and saved.get('criterion') is not None:
                    criterion.load_state_dict(saved['criterion'])
                observed = saved['observed']; running_sim = saved['running_sim']
                far_metrics = saved.get('far_metrics', {})
                if history and not far_metrics:
                    raise ValueError(
                        'Native FAR epoch resume is missing its F/G state; '
                        'restart this task with --resume-mode task'
                    )
                local_schedule, local_rows = saved['schedule_rows'], saved['epoch_rows']
                best_score = saved['best_score']; best_snapshot = saved.get('best_snapshot')
                stage_history_complete = saved.get('history_complete', True)
                recovery_notes = saved.get('recovery_notes', [])
                probe_seconds = saved.get('probe_seconds', 0.0)
                validation_seconds = saved.get('validation_seconds', 0.0)
                serialization_seconds = saved.get('serialization_seconds', 0.0)
            else:
                for key in ('optimizer_state_dict', 'scheduler_state_dict'):
                    if key not in saved and key.replace('_state_dict', '') not in saved:
                        raise ValueError(f'Legacy epoch continuation lacks {key}; use --resume-mode task')
                model.load_state_dict(compat.validated_state(saved, model, use_ema=False))
                ema_values = saved.get('ema', saved.get('ema_state_dict'))
                if ema_values is not None:
                    merged = compat.validated_state(saved, model, use_ema=True)
                    ema.load_state_dict({k: merged[k].to(device) for k,_ in model.named_parameters()})
                else:
                    ema = EMA(model, cfg.ema_decay)
                optimizer.load_state_dict(saved.get('optimizer', saved.get('optimizer_state_dict')))
                scheduler.load_state_dict(saved.get('scheduler', saved.get('scheduler_state_dict')))
                running_sim = float(saved.get('running_sim', 0.5))
                stage_history_complete = False
                recovery_notes = ['Legacy continuation: queue and selector reinitialized; '
                                  'earlier per-epoch/resource measurements may be unavailable.']
                with ema.average_parameters():
                    if history_bank is not None and history:
                        _, fresh_monitor_accuracies = (
                            history_bank.evaluate_validation(
                                model, loaders.val,
                                cfg.task_sequence[:stage + 1],
                            )
                        )
                        far_metrics = history_bank.evaluate(
                            model, history,
                            fresh_accuracies={
                                name: fresh_monitor_accuracies[name]
                                for name in history
                            },
                            tracker=tracker,
                            component_threshold_pp=far_component_threshold_pp,
                            baseline_accuracies=reference,
                        )
                    else:
                        far_metrics = {}
                    observed = {
                        name: values['a_tt']
                        for name, values in far_metrics.items()
                    }
            start_epoch = int(saved['epoch'])+1
            if start_epoch > cfg.task_epochs[task]:
                raise ValueError('Saved epoch exceeds requested task length; check --clean-epochs/--attack-epochs')
            if best_snapshot is None and best_eligible(cfg, task, int(saved['epoch'])):
                best_ref = compat.preferred(records, stage)
                if best_ref:
                    best_payload = compat.read_checkpoint(best_ref.path)
                    best_state = compat.validated_state(best_payload, model, use_ema=True)
                else:
                    with ema.average_parameters():
                        best_state = cpu_state(model)
                current_state = cpu_state(model); model.load_state_dict(best_state)
                validation = evaluate_seen(model, loaders.val, cfg.task_sequence[:stage+1], cfg, device)
                best_score = selection_mean(validation, cfg.task_sequence[:stage + 1])
                best_snapshot = {'model': best_state, 'epoch': best_ref.epoch if best_ref else saved['epoch'], 'score': best_score,
                                 'full_validation': validation}
                model.load_state_dict(current_state)
            if best_snapshot is not None:
                atomic_checkpoint(selected_path, {**best_snapshot, 'format_version': 3, 'identity': identity,
                    'settings': settings, 'stage': stage, 'task': task, 'completed': False,
                    'imported_history': legacy_origin})
        for epoch in range(start_epoch, cfg.task_epochs[task]):
            decision = selector.choose(history, reference, observed, epoch, gate,
                matched[(stage,epoch)] if matched is not None else None)
            historical_probability, regenerated_probability = (
                source_probabilities(
                    replay_source_mode,
                    far_metrics.get(decision.target),
                )
                if decision.active else (None, None)
            )
            decision_row = {'stage': stage, 'task': task, 'epoch': epoch, 'candidates': history,
                'reference_accuracy': dict(reference), 'previous_epoch_accuracy': dict(observed),
                'reference_mode': REFERENCE_MODE,
                'cumulative_reference_accuracy': {
                    name: first_learned_reference[name]
                    for name in history
                    if name in first_learned_reference
                },
                'replay_source_mode': replay_source_mode,
                'historical_probability': historical_probability,
                'regenerated_probability': regenerated_probability,
                'fg_metrics_used': copy.deepcopy(far_metrics),
                **decision.to_dict()}
            loader = epoch_loader(loaders.train.dataset, cfg, stage, epoch)
            stats, running_sim = train_epoch(model, teacher, ema, optimizer, criterion, queue, loader,
                cfg, stage, epoch, task, decision, running_sim, tracker, ablation,
                history_bank, replay_source_mode, far_metrics)
            with ema.average_parameters():
                started = time.perf_counter()
                if history_bank is not None and history:
                    validation, fresh_monitor_accuracies = (
                        history_bank.evaluate_validation(
                            model, loaders.val,
                            cfg.task_sequence[:stage + 1],
                        )
                    )
                else:
                    validation = evaluate_seen(
                        model, loaders.val,
                        cfg.task_sequence[:stage + 1], cfg, device,
                    )
                    fresh_monitor_accuracies = {}
                validation_seconds += time.perf_counter()-started
                score = selection_mean(validation, cfg.task_sequence[:stage + 1])
                if not math.isfinite(score):
                    raise FloatingPointError('Non-finite model-selection score')
                started = time.perf_counter()
                far_metrics = (
                    history_bank.evaluate(
                        model, history,
                        fresh_accuracies={
                            name: fresh_monitor_accuracies[name]
                            for name in history
                        },
                        tracker=tracker,
                        component_threshold_pp=far_component_threshold_pp,
                        baseline_accuracies=reference,
                    )
                    if history_bank is not None and history else {}
                )
                observed = {
                    name: values['a_tt']
                    for name, values in far_metrics.items()
                }
                probe_seconds += time.perf_counter()-started
                if best_eligible(cfg, task, epoch) and score > best_score:
                    best_score = score
                    best_snapshot = {'model': cpu_state(model), 'epoch': epoch, 'score': score, 'full_validation': validation}
                    atomic_checkpoint(selected_path, {**best_snapshot, 'format_version': 3, 'identity': identity,
                        'settings': settings, 'stage': stage, 'task': task, 'completed': False,
                        'imported_history': legacy_origin})
            scheduler.step(); local_schedule.append(decision_row)
            local_rows.append({'stage': stage, 'task': task, 'epoch': epoch, 'replay_target': decision.target,
                'replay_source_mode': replay_source_mode,
                'best_eligible': best_eligible(cfg, task, epoch),
                'warmup_epochs': warmup_epochs(cfg, task),
                'reference_mode': REFERENCE_MODE,
                'stage_reference_accuracy': dict(reference),
                'cumulative_reference_accuracy': {
                    name: first_learned_reference[name]
                    for name in history
                    if name in first_learned_reference
                },
                'selection_score': score, 'validation': validation,
                'fg_monitor': copy.deepcopy(far_metrics), **stats})
            payload = {'format_version': 3, 'identity': identity, 'settings': settings, 'stage': stage,
                'imported_history': legacy_origin,
                'task': task, 'planned_epochs': cfg.task_epochs[task], 'completed': False,
                'epoch': epoch, 'model': cpu_state(model), 'teacher': cpu_state(teacher) if teacher else None,
                'ema': {k:v.detach().cpu().clone() for k,v in ema.state_dict().items()},
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'criterion': criterion.state_dict() if criterion is not None else None,
                'queue': queue.state_dict() if queue is not None else None,
                'selector': selector.state_dict(), 'resources': tracker.state_dict(),
                'reference': reference, 'observed': observed, 'running_sim': running_sim,
                'far_metrics': copy.deepcopy(far_metrics),
                'far_history_summary': history_bank.summary() if history_bank is not None else None,
                'far_history_state': history_bank.state_dict() if history_bank is not None else None,
                'first_learned_reference': dict(first_learned_reference),
                'reference_sources': copy.deepcopy(reference_sources), 'best_score': best_score,
                'best_snapshot': best_snapshot, 'schedule_rows': local_schedule, 'epoch_rows': local_rows,
                'history_complete': stage_history_complete, 'recovery_notes': recovery_notes,
                'probe_seconds': probe_seconds, 'validation_seconds': validation_seconds,
                'serialization_seconds': serialization_seconds}
            started = time.perf_counter(); atomic_checkpoint(resume_path, payload)
            serialization_seconds += time.perf_counter()-started
            atomic_json(cfg.run_dir/'replay_schedule.json', {**identity, 'policy': policy, 'gate': gate,
                'replay_source_mode': replay_source_mode,
                'reference_mode': REFERENCE_MODE,
                'dtr_drop_basis': 'previous_stage_best',
                'complete': False, 'history_complete': history_complete and stage_history_complete,
                'epochs': schedule_rows+local_schedule})
            atomic_json(cfg.run_dir/'epoch_metrics.json', stage_rows+local_rows)
            if history_bank is not None:
                atomic_json(cfg.run_dir/'far_history_summary.json', history_bank.summary())
                atomic_json(cfg.run_dir/'fg_decomposition.json', [
                    {'stage': row['stage'], 'task': row['task'], 'epoch': row['epoch'],
                     'metrics': row.get('fg_monitor', {})}
                    for row in stage_rows + local_rows
                ])
            print(f'{cfg.method} task={task} epoch={epoch+1}/{cfg.task_epochs[task]} '
                  f'replay={decision.target} loss={stats["loss"]:.4f} val={score:.2f}', flush=True)
        if best_snapshot is None:
            raise RuntimeError('No selected model; task length must be positive')
        model.load_state_dict(best_snapshot['model'])
        if history_bank is not None:
            first_learned_reference[task] = history_bank.establish_task(
                task, model, selected_path.name, tracker=tracker,
                build_training=(
                    eager_training_cache(replay_source_mode)
                    and stage < len(cfg.task_sequence) - 1
                ),
            )
            atomic_json(cfg.run_dir/'far_history_summary.json', history_bank.summary())
        else:
            started = time.perf_counter()
            first_learned_reference[task] = evaluate_seen(
                model, loaders.selection_val, [task], cfg, device
            )[task]
            probe_seconds += time.perf_counter()-started
        reference_sources[task] = {'stage': stage, 'selected_epoch': best_snapshot['epoch'],
            'checkpoint': selected_path.name,
            'split': 'far_fixed_monitor' if history_bank is not None else 'selection_val',
            'origin': 'native'}
        queue_bytes = (
            tensor_storage_bytes(queue.features, queue.labels)
            if queue is not None else 0
        )
        history_bytes = history_bank.storage_bytes() if history_bank is not None else 0
        resource = tracker.finish(queue_bytes + history_bytes)
        resource.update({'replay_examples': sum(r['replay_examples'] for r in local_rows),
            'clean_replay_examples': sum(r['clean_replay_examples'] for r in local_rows),
            'replay_batches': sum(r['replay_batches'] for r in local_rows),
            'historical_replay_examples': sum(r['historical_replay_examples'] for r in local_rows),
            'regenerated_replay_examples': sum(r['regenerated_replay_examples'] for r in local_rows),
            'historical_replay_batches': sum(r['historical_replay_batches'] for r in local_rows),
            'regenerated_replay_batches': sum(r['regenerated_replay_batches'] for r in local_rows),
            'lazy_cache_builds': sum(r['lazy_cache_builds'] for r in local_rows),
            'far_history_storage_bytes': history_bytes,
            'replay_source_mode': replay_source_mode,
            'active_replay_epochs': sum(int(r['active']) for r in local_schedule),
            'selection_probe_wall_seconds': probe_seconds, 'checkpoint_validation_wall_seconds': validation_seconds,
            'resume_serialization_wall_seconds': serialization_seconds, 'provenance': 'measured'})
        if not stage_history_complete:
            resource['continuation_measurements'] = dict(resource)
            for key in list(resource):
                if key.endswith(('_seconds', '_bytes', '_samples')) or key in (
                    'replay_examples','clean_replay_examples','replay_batches','active_replay_epochs'):
                    resource[key] = None
            resource['provenance'] = 'partial_legacy_continuation; full_stage_cost_unknown'
        upsert_resource_row(cfg.run_dir, resource)
        matrix_row = {'stage': stage, 'task': task, 'selected_epoch': best_snapshot['epoch'],
            'accuracies': best_snapshot['full_validation'], 'split': 'full_validation', 'provenance': 'measured'}
        accuracy_matrix.append(matrix_row)
        summary = {'identity': identity, 'schedule_rows': local_schedule, 'epoch_rows': local_rows,
            'accuracy_matrix_row': matrix_row, 'resource_row': resource, 'history_complete': stage_history_complete,
            'completed': True, 'recovery_notes': recovery_notes}
        atomic_checkpoint(selected_path, {**best_snapshot, 'format_version': 3, 'identity': identity,
            'imported_history': legacy_origin,
            'settings': settings, 'stage': stage, 'task': task, 'planned_epochs': cfg.task_epochs[task],
            'completed': True, 'first_learned_reference': dict(first_learned_reference),
            'reference_sources': copy.deepcopy(reference_sources), 'stage_summary': summary})
        atomic_json(completed_path, summary)
        schedule_rows.extend(local_schedule); stage_rows.extend(local_rows)
        history_complete &= stage_history_complete
        saved = None
        # state_dict() retains optimizer momentum tensors even after deleting
        # the optimizer. The resume payload is already safely on disk.
        payload = None
        del teacher, queue, criterion, ema, optimizer, scheduler, tracker
    # With only a final-task checkpoint, skipped history remains explicitly missing.
    if final_only:
        final_ref = compat.preferred(records, len(cfg.task_sequence)-1)
        model.load_state_dict(compat.validated_state(compat.boundary_payload(
            compat.read_checkpoint(final_ref.path), final_ref), model, use_ema=True))
    atomic_json(cfg.run_dir/'replay_schedule.json', {**identity, 'policy': policy, 'gate': gate,
        'replay_source_mode': replay_source_mode,
        'reference_mode': REFERENCE_MODE,
        'dtr_drop_basis': 'previous_stage_best',
        'complete': True, 'history_complete': history_complete, 'epochs': schedule_rows})
    atomic_json(cfg.run_dir/'epoch_metrics.json', stage_rows)
    atomic_json(cfg.run_dir/'accuracy_matrix.json', accuracy_matrix)
    atomic_json(cfg.run_dir/'replay_references.json', {
        'mode': 'initial_task_best_checkpoint',
        'dtr_reference_mode': REFERENCE_MODE,
        'accuracies': first_learned_reference,
        'initial_task_reference_accuracies': first_learned_reference,
        'stage_references_are_recorded_in': 'replay_schedule.json and epoch_metrics.json',
        'sources': reference_sources, 'probe_size': cfg.selection_probe_size, 'smoke': smoke,
        'missing_tasks': [t for t in cfg.task_sequence if t not in first_learned_reference]})
    if history_bank is not None:
        atomic_json(cfg.run_dir/'far_history_summary.json', history_bank.summary())
        atomic_json(cfg.run_dir/'fg_decomposition.json', [
            {'stage': row['stage'], 'task': row['task'], 'epoch': row['epoch'],
             'metrics': row.get('fg_monitor', {})}
            for row in stage_rows
        ])
    # Training-only state is no longer needed; checkpoints/caches stay on disk.
    history_bank = previous_state = saved = None
    prepare_final_evaluation(model, device)
    # Save each evaluation condition independently, allowing AA interruptions to
    # continue without re-running already measured conditions.
    evaluation_settings = {'evaluation_seed': cfg.evaluation_seed,
        'autoattack_sampling_policy': AUTOATTACK_SAMPLING_POLICY,
        'subset_seed': cfg.autoattack_subset_seed, 'autoattack_samples': cfg.autoattack_samples,
        'test_dataset_size': len(loaders.test.dataset),
        'autoattack_samples_effective': (len(loaders.test.dataset) if cfg.autoattack_samples == 0
                                       else min(cfg.autoattack_samples, len(loaders.test.dataset))),
        'autoattack_batch_size': cfg.autoattack_batch_size, 'requested_methods': cfg.final_test_methods,
        'eval_attack_steps': cfg.eval_attack_steps, 'cw_eval_steps': cfg.cw_eval_steps,
        'linf_eps': cfg.linf_eps, 'l2_eps': cfg.l2_eps, 'patch_size': cfg.patch_size}
    final_fingerprint = digest({'model': model_digest(model),
        'autoattack_sampling_policy': AUTOATTACK_SAMPLING_POLICY,
        'evaluation_seed': cfg.evaluation_seed, 'subset_seed': cfg.autoattack_subset_seed,
        'autoattack_samples': cfg.autoattack_samples, 'autoattack_batch_size': cfg.autoattack_batch_size,
        'split_signature': split_signature, 'settings': settings, 'data_root': str(Path(cfg.data_root).resolve())})
    meta = compat.json_read(cfg.run_dir/'final_test_meta.json', {})
    results = compat.json_read(cfg.run_dir/'final_test.json', {}) if meta.get('fingerprint') == final_fingerprint else {}
    completion_marker = cfg.run_dir/'run_complete.json'
    if completion_marker.exists():
        completion_marker.unlink()
    atomic_json(cfg.run_dir/'final_test_meta.json', {'fingerprint': final_fingerprint,
        'evaluation_settings': evaluation_settings, 'complete': False})
    from pace_aligned.common import evaluate_autoattack, evaluate_accuracy
    feature_store = None
    if cfg.plot_final_features:
        from pace_aligned.feature_plots import make_feature_store
        feature_store = make_feature_store(model, loaders.test, cfg)
    evaluation_methods = feature_store.methods if feature_store is not None else cfg.final_test_methods
    for method in evaluation_methods:
        if feature_store is not None and feature_store.has(method):
            results[method] = feature_store.accuracy(method)
            atomic_json(cfg.run_dir/'final_test.json', results)
            continue
        if method in results and feature_store is None:
            continue
        # Old accuracy-only JSON cannot reconstruct features. Re-evaluate a
        # missing condition once, then reuse its CPU feature cache on resume.
        extra = {'feature_collector': feature_store.collector(method)} if feature_store is not None else {}
        if method in ('autoattack_l2','autoattack'):
            norm, epsilon = ('L2',cfg.l2_eps) if method == 'autoattack_l2' else ('Linf',cfg.linf_eps)
            results[method] = evaluate_autoattack(model, loaders.test, norm, epsilon, cfg, device, **extra)
        elif feature_store is not None:
            results[method] = evaluate_accuracy(model, loaders.test, method, cfg, device, **extra)
        else:
            results.update(evaluate_seen(model, loaders.test, [method], cfg, device))
        atomic_json(cfg.run_dir/'final_test.json', results)
    atomic_json(cfg.run_dir/'final_test_meta.json', {'fingerprint': final_fingerprint, 'complete': True,
        'requested_methods': cfg.final_test_methods, 'evaluation_settings': evaluation_settings})
    if feature_store is not None:
        feature_store.render(cfg)
    atomic_json(cfg.run_dir/'run_complete.json', {'identity': identity, 'status': 'smoke_test' if smoke else 'experiment',
        'ablation': ablation.name, 'final_test': results, 'history_complete': history_complete,
        'imported_history': legacy_origin, 'training_complete': True,
        'replay_source_mode': replay_source_mode})
    from export_run_artifacts import export
    export(cfg.run_dir, plots=not smoke)
    print('Completed:', cfg.run_dir, flush=True)
    return cfg.run_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='PACE with four historical/re-optimized replay-source modes'
    )
    add_backbone_argument(p)
    from pace_aligned.feature_plots import add_feature_arguments
    add_feature_arguments(p)
    p.add_argument('--policy', choices=POLICIES, default='dtr')
    p.add_argument(
        '--replay-source-mode', choices=SOURCE_MODES,
        default='fg_adaptive_lazy',
        help=('historical=all cached; regenerated=all re-attacked; '
              'fg_adaptive=F/G routing with eager caches; '
              'fg_adaptive_lazy=F/G routing with cache-on-first-use'),
    )
    p.add_argument('--far-monitor-size', type=int, default=500)
    p.add_argument('--far-history-size', type=int, default=500)
    p.add_argument('--far-component-threshold-pp', type=float, default=0.0)
    p.add_argument('--far-cache-dtype', choices=['float32', 'uint8'], default='float32')
    p.add_argument('--ablation', choices=ABLATIONS, default='full')
    p.add_argument('--task-sequence', nargs=4,
                   choices=('clean', 'patch', 'pgd_l2', 'pgd'),
                   default=('clean', 'patch', 'pgd_l2', 'pgd'),
                   help='Four distinct stages, clean first; pgd means PGD-Linf. '
                        'Default preserves the original Clean-Patch-L2-Linf order.')
    p.add_argument('--dataset', choices=['cifar10','cifar100','tiny_imagenet','imagenette'], default='cifar100')
    p.add_argument('--data-root', default='./data'); p.add_argument('--output-root', default='./runs_replay')
    p.add_argument('--run-name', default='pace_v3'); p.add_argument('--run-dir')
    p.add_argument('--seed', type=int, default=42); p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--evaluation-seed', type=int, default=42); p.add_argument('--autoattack-subset-seed', type=int, default=42)
    p.add_argument('--autoattack-batch-size', type=int, default=None,
                   help='Final AutoAttack batch only; does not change training batch size')
    p.add_argument('--autoattack-samples', type=int, default=1000,
                   help='Class-balanced AutoAttack subset size; 0 (or >= test size) uses the full test set')
    p.add_argument('--batch-size', type=int); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--depth', type=int, default=34, help='WRN only; ignored by ResNet-18')
    p.add_argument('--widen-factor', type=int, default=10, help='WRN only; ignored by ResNet-18')
    p.add_argument('--dropout', type=float, default=0.3, help='WRN only; ResNet-18 has no dropout')
    p.add_argument('--clean-epochs', type=int, default=100); p.add_argument('--attack-epochs', type=int, default=30)
    p.add_argument('--weight-floor', type=float, default=0.2); p.add_argument('--queue-capacity', type=int, default=256)
    p.add_argument('--exclude-clean', action='store_true'); p.add_argument('--drop-threshold-pp', type=float, default=1.0)
    p.add_argument('--match-schedule', type=Path)
    p.add_argument('--resume', nargs='?', const='auto', metavar='PATH', help='Scan run folder, or import a checkpoint directory/file')
    p.add_argument('--start-task', help='Task name or 0/1/2/3 in the selected --task-sequence')
    p.add_argument('--resume-mode', choices=['task','epoch'], default='task')
    p.add_argument('--completed-through', help='Declare a completed boundary for best-only legacy checkpoints')
    p.add_argument('--checkpoint-task', help='Task of an explicitly named checkpoint with an arbitrary filename')
    p.add_argument('--eval-only', action='store_true'); p.add_argument('--inspect-checkpoints', action='store_true')
    p.add_argument('--skip-autoattack', action='store_true'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--dry-run', action='store_true',
                   help='Print resolved settings and task order without loading data or training')
    args = p.parse_args(argv)
    if args.task_sequence[0] != 'clean' or len(set(args.task_sequence)) != 4:
        p.error('--task-sequence must contain clean, pgd, pgd_l2 and patch exactly once, with clean first')
    return args


def main(argv=None):
    a = parse_args(argv)
    if min(a.clean_epochs, a.attack_epochs) < 1:
        raise ValueError('Epoch counts must be positive')
    batch_size = a.batch_size or {'cifar10':64,'cifar100':64,'tiny_imagenet':16,'imagenette':64}[a.dataset]
    cfg = ExperimentConfig(
        method=f'{method_name(a.policy,a.ablation)}_{a.replay_source_mode}',
        backbone=a.backbone,
        dataset=a.dataset,
        data_root=a.data_root, output_root=a.output_root, run_name=a.run_name, run_directory=a.run_dir,
        seed=a.seed, split_seed=a.split_seed, evaluation_seed=a.evaluation_seed,
        autoattack_subset_seed=a.autoattack_subset_seed, batch_size=batch_size,
        val_size=1000 if a.dataset == 'imagenette' else 5000,
        selection_probe_size=a.far_monitor_size,
        workers=a.workers, device=a.device,
        resume=bool(a.resume), depth=a.depth, widen_factor=a.widen_factor, dropout=a.dropout,
        task_sequence=tuple(a.task_sequence),
        task_epochs={'clean':a.clean_epochs,'patch':a.attack_epochs,'pgd_l2':a.attack_epochs,'pgd':a.attack_epochs})
    if a.far_monitor_size > cfg.val_size:
        raise ValueError('--far-monitor-size cannot exceed the validation-set size')
    cfg.patch_size = cfg.image_size//4; cfg.l2_eps = 0.5*cfg.image_size/32; cfg.l2_alpha = 0.1*cfg.image_size/32
    cfg.autoattack_batch_size = min(32,batch_size)
    if a.autoattack_batch_size is not None:
        if a.autoattack_batch_size <= 0:
            raise ValueError('--autoattack-batch-size must be positive')
        cfg.autoattack_batch_size = a.autoattack_batch_size
    if a.autoattack_samples < 0:
        raise ValueError('--autoattack-samples must be nonnegative (0 = full test set)')
    cfg.autoattack_samples = a.autoattack_samples
    from pace_aligned.feature_plots import apply_feature_arguments
    apply_feature_arguments(cfg, a)
    if a.skip_autoattack or a.smoke:
        cfg.final_test_methods = tuple(x for x in cfg.final_test_methods if not x.startswith('autoattack'))
    if a.smoke:
        if a.dataset not in ('cifar10','cifar100'):
            raise ValueError('Smoke test supports CIFAR-shaped inputs')
        cfg.device,cfg.workers,cfg.batch_size = 'cpu',0,4
        cfg.depth,cfg.widen_factor = 10,1; cfg.task_epochs = {t:3 for t in cfg.task_sequence}
        cfg.train_attack_steps = cfg.eval_attack_steps = cfg.cw_eval_steps = 1
        cfg.max_train_batches = cfg.max_eval_batches = 1
        torch.set_num_threads(2)
    if a.dry_run:
        print(json.dumps({'config': asdict(cfg), 'run_dir': str(cfg.run_dir),
            'task_sequence': list(cfg.task_sequence),
            'settings': protocol_settings(cfg, not a.exclude_clean, a.drop_threshold_pp,
                a.smoke, AblationConfig(a.ablation, a.weight_floor, a.queue_capacity),
                a.replay_source_mode, a.far_monitor_size, a.far_history_size,
                a.far_component_threshold_pp, a.far_cache_dtype)}, indent=2))
        return
    run_experiment(cfg, a.policy, not a.exclude_clean, a.drop_threshold_pp, a.match_schedule, a.smoke,
        AblationConfig(a.ablation,a.weight_floor,a.queue_capacity),
        None if a.resume in (None,'auto') else a.resume, a.start_task, a.resume_mode,
        a.completed_through, a.checkpoint_task, a.eval_only, a.inspect_checkpoints,
        a.replay_source_mode, a.far_monitor_size, a.far_history_size,
        a.far_component_threshold_pp, a.far_cache_dtype)


if __name__ == '__main__':
    main()

"""One explicit preset per ablation; all other settings remain shared."""
from dataclasses import asdict, dataclass

ABLATIONS = ('full', 'only_cls', 'no_distillation', 'no_contrast',
             'fixed_weights', 'no_queue', 'fixed_temperature', 'no_dtr')
LABELS = {'full': 'PACE', 'only_cls': 'Only L_cls', 'no_distillation': 'No distillation',
          'no_contrast': 'No contrast term', 'fixed_weights': 'Fixed weights',
          'no_queue': 'No queue', 'fixed_temperature': 'Fixed temperature', 'no_dtr': 'No DTR'}

@dataclass(frozen=True)
class AblationConfig:
    name: str = 'full'
    weight_floor: float = 0.2
    queue_capacity: int = 256

    def __post_init__(self):
        if self.name not in ABLATIONS:
            raise ValueError(f'Unknown ablation: {self.name}')
        if not 0 <= self.weight_floor <= 1:
            raise ValueError('weight_floor must be in [0, 1]')
        if self.queue_capacity <= 0:
            raise ValueError('queue_capacity must be positive; use no_queue to disable it')

    @property
    def feature_loss(self):
        return 'none' if self.name == 'only_cls' else 'cosine' if self.name == 'no_contrast' else 'contrastive'

    @property
    def key_source(self):
        return 'student' if self.name == 'no_distillation' else 'teacher'

    @property
    def use_teacher(self):
        return self.feature_loss != 'none' and self.key_source == 'teacher'

    @property
    def use_queue(self):
        return self.feature_loss == 'contrastive' and self.name != 'no_queue'

    @property
    def adaptive_temperature(self):
        return self.feature_loss == 'contrastive' and self.name != 'fixed_temperature'

    @property
    def gradient_weighting(self):
        return self.feature_loss != 'none' and self.name != 'fixed_weights'

    @property
    def replay(self):
        return self.name not in ('no_dtr', 'only_cls')

    def to_dict(self):
        return {**asdict(self), **{k: getattr(self, k) for k in (
            'feature_loss', 'key_source', 'use_teacher', 'use_queue',
            'adaptive_temperature', 'gradient_weighting', 'replay')}}

def method_name(policy, ablation='full'):
    return f'pace_{policy}' + ('' if ablation == 'full' else f'_{ablation}')

def make_criterion(ablation, task):
    if task == 'clean' or ablation.feature_loss == 'none':
        return None
    from pace_components import PureFeatureDistillationLoss, AdaptiveDecoupledLACDLoss, DecoupledLACDLoss
    if ablation.feature_loss == 'cosine':
        return PureFeatureDistillationLoss()
    if ablation.adaptive_temperature:
        return AdaptiveDecoupledLACDLoss()
    return DecoupledLACDLoss(temp_pos=0.1, temp_neg=0.1)

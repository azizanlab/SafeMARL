from safe_marl.algorithms.critics.v_critic import VCritic
from safe_marl.algorithms.critics.soft_twin_continuous_q_critic import SoftTwinContinuousQCritic

CRITIC_REGISTRY = {
    "happo": VCritic,
    "hasac": SoftTwinContinuousQCritic,
}

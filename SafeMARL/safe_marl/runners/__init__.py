from safe_marl.runners.on_policy_ha_runner import OnPolicyHARunner
from safe_marl.runners.off_policy_ha_runner import OffPolicyHARunner
from safe_marl.runners.madac_runner import MADACRunner

RUNNER_REGISTRY = {
    "happo": OnPolicyHARunner,
    "hasac": OffPolicyHARunner,
    "madac": MADACRunner,
}

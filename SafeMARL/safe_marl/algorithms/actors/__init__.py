from safe_marl.algorithms.actors.happo import HAPPO
from safe_marl.algorithms.actors.hasac import HASAC

ALGO_REGISTRY = {
    "happo": HAPPO,
    "hasac": HASAC,
}

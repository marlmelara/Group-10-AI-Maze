from typing import Tuple

Position = Tuple[int, int]

START_POS: Position = (0, 31)
GOAL_POS: Position = (63, 32)

HAZARD_SEED: int = 4368
HAZARD_DENSITY: float = 0.07

PIT_RATIO: float = 0.60
CONFUSION_RATIO: float = 0.20
TELEPORT_RATIO: float = 0.20
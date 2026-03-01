from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ComponentConfig:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerConfig(ComponentConfig):
    pass

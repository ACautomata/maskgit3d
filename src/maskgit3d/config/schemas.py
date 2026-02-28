from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class ComponentConfig:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerConfig(ComponentConfig):
    pass

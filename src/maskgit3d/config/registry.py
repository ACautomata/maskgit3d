from typing import Any, Callable, Dict

Registry = Dict[str, Callable[..., Any]]


def resolve_component(registry: Registry, name: str):
    if name not in registry:
        raise ValueError(f"Unknown component type: {name}. Available: {list(registry.keys())}")
    return registry[name]

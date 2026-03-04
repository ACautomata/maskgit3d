"""Tests for config/registry.py"""

import pytest

from maskgit3d.config.registry import resolve_component


def test_resolve_component_success():
    """Test successful component resolution."""
    registry = {"model_a": lambda: "model_a_instance", "model_b": lambda: "model_b_instance"}
    result = resolve_component(registry, "model_a")
    assert result() == "model_a_instance"


def test_resolve_component_another_key():
    """Test resolution with different keys."""
    registry = {"encoder": lambda x: x * 2, "decoder": lambda x: x // 2}
    encoder = resolve_component(registry, "encoder")
    assert encoder(5) == 10


def test_resolve_component_not_found():
    """Test that ValueError is raised for unknown components."""
    registry = {"model_a": lambda: "a", "model_b": lambda: "b"}
    with pytest.raises(ValueError) as exc_info:
        resolve_component(registry, "unknown_model")
    assert "Unknown component type" in str(exc_info.value)
    assert "unknown_model" in str(exc_info.value)
    assert "model_a" in str(exc_info.value)
    assert "model_b" in str(exc_info.value)


def test_resolve_component_empty_registry():
    """Test resolution from empty registry."""
    registry = {}
    with pytest.raises(ValueError) as exc_info:
        resolve_component(registry, "any_key")
    assert "Unknown component type" in str(exc_info.value)
    assert "Available: []" in str(exc_info.value)

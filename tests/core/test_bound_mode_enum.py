import pytest
from pyMOFL.core.bound_mode_enum import BoundModeEnum

def test_bound_mode_enum_members():
    assert BoundModeEnum.INITIALIZATION.name == "INITIALIZATION"
    assert BoundModeEnum.OPERATIONAL.name == "OPERATIONAL"
    assert isinstance(BoundModeEnum.INITIALIZATION.value, int)
    assert isinstance(BoundModeEnum.OPERATIONAL.value, int)

def test_bound_mode_enum_docstring():
    assert BoundModeEnum.__doc__ is not None
    assert "mode in which bounds are used" in BoundModeEnum.__doc__ 
import pytest
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum

def test_quantization_type_enum_members():
    assert QuantizationTypeEnum.CONTINUOUS.name == "CONTINUOUS"
    assert QuantizationTypeEnum.INTEGER.name == "INTEGER"
    assert QuantizationTypeEnum.STEP.name == "STEP"
    assert isinstance(QuantizationTypeEnum.CONTINUOUS.value, int)
    assert isinstance(QuantizationTypeEnum.INTEGER.value, int)
    assert isinstance(QuantizationTypeEnum.STEP.value, int)

def test_quantization_type_enum_docstring():
    assert QuantizationTypeEnum.__doc__ is not None
    assert "quantization type of variables" in QuantizationTypeEnum.__doc__ 
from dataclasses import dataclass

import numpy as np


@dataclass
class ToleranceConfig:
    strict_atol: float = 1e-6
    strict_rtol: float = 1e-9

    large_threshold: float = 1e9
    large_atol: float = 1e-3
    large_rtol: float = 1e-8

    medium_threshold: float = 1e6
    medium_atol: float = 1e5
    medium_rtol: float = 1e-4

    small_threshold: float = 1e4
    small_atol: float = 1.0
    small_rtol: float = 1e-5

    noise_rtol: float = 0.5
    noise_atol: float = 1e6
    noise_magnitude_min: float = 0.1
    noise_magnitude_max: float = 10.0


class ToleranceChecker:
    def __init__(self, config: ToleranceConfig | None = None) -> None:
        self.cfg = config or ToleranceConfig()

    def check(self, expected: float, actual: float, is_noisy: bool, is_optimal: bool) -> bool:
        if np.isnan(expected) and np.isnan(actual):
            return True
        if np.isnan(expected) or np.isnan(actual):
            return False
        if is_noisy and not is_optimal:
            return self._check_noisy(expected, actual)
        return self._check_deterministic(expected, actual)

    def _check_noisy(self, expected: float, actual: float) -> bool:
        if abs(expected) > 1e3:
            rel = abs((actual - expected) / expected) if expected != 0 else float("inf")
            passed = rel <= self.cfg.noise_rtol
        else:
            passed = abs(actual - expected) <= self.cfg.noise_atol
        if expected != 0 and actual != 0:
            ratio = abs(actual / expected)
            passed = passed and (
                self.cfg.noise_magnitude_min <= ratio <= self.cfg.noise_magnitude_max
            )
        return passed

    def _check_deterministic(self, expected: float, actual: float) -> bool:
        ae = abs(expected)
        if ae > self.cfg.large_threshold:
            atol, rtol = self.cfg.large_atol, self.cfg.large_rtol
        elif ae > self.cfg.medium_threshold:
            atol, rtol = self.cfg.medium_atol, self.cfg.medium_rtol
        elif ae > self.cfg.small_threshold:
            atol, rtol = self.cfg.small_atol, self.cfg.small_rtol
        else:
            atol, rtol = self.cfg.strict_atol, self.cfg.strict_rtol
        return bool(np.isclose(actual, expected, atol=atol, rtol=rtol))

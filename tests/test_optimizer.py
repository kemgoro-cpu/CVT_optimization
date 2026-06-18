import numpy as np

from cvt_optimizer.optimizer import (
    OptimizationOptions,
    apply_map_update,
    enforce_monotonic_map,
    isotonic_increasing,
)


def test_isotonic_increasing_uses_least_squares_block_average() -> None:
    result = isotonic_increasing(np.array([1000.0, 1800.0, 1400.0, 2200.0]))

    np.testing.assert_allclose(result, [1000.0, 1600.0, 1600.0, 2200.0])


def test_enforce_monotonic_map_in_both_directions() -> None:
    values = np.array(
        [
            [1000.0, 1700.0, 1500.0],
            [900.0, 1600.0, 1400.0],
            [1300.0, 1200.0, 2000.0],
        ]
    )

    result = enforce_monotonic_map(
        values,
        speed_direction=True,
        throttle_direction=True,
    )

    assert np.all(np.diff(result, axis=1) >= -1e-8)
    assert np.all(np.diff(result, axis=0) >= -1e-8)


def test_apply_map_update_respects_selected_monotonic_direction() -> None:
    current = np.array([[1000.0, 1200.0, 1400.0], [1300.0, 1500.0, 1700.0]])
    target = np.array([[1000.0, 1800.0, 1300.0], [1600.0, 1400.0, 1900.0]])
    coverage = np.ones_like(current, dtype=int)

    result = apply_map_update(
        current,
        target,
        coverage,
        OptimizationOptions(monotonic_speed=True),
    )

    assert np.all(np.diff(result, axis=1) >= -1e-8)

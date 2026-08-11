import numpy as np
import pytest

from formation_control.experiment import (
    FIELD_SLICES,
    FIELD_WIDTHS,
    FormationExperimentHistory,
    SNAPSHOT_SCHEMA_VERSION,
    SNAPSHOT_SIZE,
    field_names,
    pack_snapshot,
    unpack_snapshot,
)


def test_snapshot_schema_is_contiguous_and_versioned():
    cursor = 0
    for name in field_names():
        width = FIELD_WIDTHS[name]
        field_slice = FIELD_SLICES[name]
        assert field_slice.start == cursor
        assert field_slice.stop == cursor + width
        cursor += width
    assert cursor == SNAPSHOT_SIZE

    decoded = unpack_snapshot(pack_snapshot({"role": 0.0}))
    assert decoded["schema_version"] == pytest.approx(SNAPSHOT_SCHEMA_VERSION)


def test_snapshot_round_trip_scalar_and_vectors():
    values = {
        "role": 0.0,
        "fallback": 1.0,
        "controller_time_s": 0.003,
        "desired_relative_position": np.array([0.0, -1.8, 0.0]),
        "thruster_forces": np.linspace(-4.0, 3.0, 8),
        "conservative_values": np.array([1.0, 2.0, 3.0, 4.0]),
        "generalized_velocity": np.arange(6.0),
    }
    decoded = unpack_snapshot(pack_snapshot(values))

    for name, expected in values.items():
        np.testing.assert_allclose(decoded[name], expected)


def test_snapshot_missing_fields_are_nan():
    decoded = unpack_snapshot(pack_snapshot({"role": 1.0}))
    assert np.isnan(decoded["slack"])
    assert np.all(np.isnan(decoded["image_coordinates"]))


def test_snapshot_rejects_unknown_or_wrong_width_fields():
    with pytest.raises(KeyError):
        pack_snapshot({"not_a_field": 1.0})

    with pytest.raises(ValueError):
        pack_snapshot({"thruster_forces": np.zeros(7)})


def test_history_npz_round_trip(tmp_path):
    history = FormationExperimentHistory(
        arrays={
            "times": np.array([0.0, 0.02]),
            "positions": np.zeros((2, 2, 3)),
        },
        metadata={
            "robots": ["leader", "follower"],
            "edges": [{"observer": "follower", "target": "leader"}],
        },
    )
    path = history.save(tmp_path / "history.npz")
    loaded = FormationExperimentHistory.load(path)

    assert loaded.metadata == history.metadata
    np.testing.assert_allclose(loaded.arrays["times"], history.arrays["times"])
    np.testing.assert_allclose(
        loaded.arrays["positions"],
        history.arrays["positions"],
    )

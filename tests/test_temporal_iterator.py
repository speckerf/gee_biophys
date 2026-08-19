from datetime import UTC, datetime

from gee_biophys.config import Temporal


def test_fixed_monthly_interval_generation():
    """Monthly fixed cadence should generate one interval per month."""
    data = {
        "start": "2023-01-01",
        "end": "2023-12-31",
        "cadence": {"type": "fixed", "interval": "monthly"},
    }
    t = Temporal(**data)

    assert t.start == datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)
    assert t.end == datetime(2023, 12, 31, 0, 0, 0, tzinfo=UTC)
    assert len(list(t.iter_date_ranges())) == 12


def test_fixed_bimonthly_interval_generation():
    """Bimonthly fixed cadence should generate two intervals for the window."""
    data = {
        "start": "2023-01-15",
        "end": "2023-04-14",
        "cadence": {"type": "fixed", "interval": "bimonthly"},
    }
    t = Temporal(**data)

    assert len(list(t.iter_date_ranges())) == 2


def test_seasonal_in_year_interval_generation():
    """Seasonal cadence should yield one window per year for in-year seasons."""
    data = {
        "start": "2021-01-01",
        "end": "2023-12-31",
        "cadence": {"type": "seasons", "start": "05-15", "end": "09-15"},
    }
    t = Temporal(**data)

    ranges = list(t.iter_date_ranges())
    assert len(ranges) == 3
    assert ranges[0] == (
        datetime(2021, 5, 15, 0, 0, 0, tzinfo=UTC),
        datetime(2021, 9, 15, 0, 0, 0, tzinfo=UTC),
    )
    assert ranges[1] == (
        datetime(2022, 5, 15, 0, 0, 0, tzinfo=UTC),
        datetime(2022, 9, 15, 0, 0, 0, tzinfo=UTC),
    )
    assert ranges[2] == (
        datetime(2023, 5, 15, 0, 0, 0, tzinfo=UTC),
        datetime(2023, 9, 15, 0, 0, 0, tzinfo=UTC),
    )


def test_seasonal_cross_year_interval_generation():
    """Seasonal cadence should support wrap-around seasons across calendar years."""
    data = {
        "start": "2020-01-01",
        "end": "2022-12-31",
        "cadence": {"type": "seasons", "start": "11-01", "end": "03-01"},
    }
    t = Temporal(**data)

    ranges = list(t.iter_date_ranges())
    assert len(ranges) == 4
    assert ranges[0] == (
        datetime(2020, 1, 1, 0, 0, 0, tzinfo=UTC),
        datetime(2020, 3, 1, 0, 0, 0, tzinfo=UTC),
    )
    assert ranges[1] == (
        datetime(2020, 11, 1, 0, 0, 0, tzinfo=UTC),
        datetime(2021, 3, 1, 0, 0, 0, tzinfo=UTC),
    )
    assert ranges[2] == (
        datetime(2021, 11, 1, 0, 0, 0, tzinfo=UTC),
        datetime(2022, 3, 1, 0, 0, 0, tzinfo=UTC),
    )
    assert ranges[3] == (
        datetime(2022, 11, 1, 0, 0, 0, tzinfo=UTC),
        datetime(2022, 12, 31, 0, 0, 0, tzinfo=UTC),
    )


def test_fixed_dekadal_interval_generation():
    """Dekadal cadence should reset each month and split dates into 10-day windows."""
    data = {
        "start": "2024-02-01",
        "end": "2024-03-01",
        "cadence": {"type": "fixed", "interval": "dekadal"},
    }
    t = Temporal(**data)

    assert list(t.iter_date_ranges()) == [
        (
            datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC),
            datetime(2024, 2, 11, 0, 0, 0, tzinfo=UTC),
        ),
        (
            datetime(2024, 2, 11, 0, 0, 0, tzinfo=UTC),
            datetime(2024, 2, 21, 0, 0, 0, tzinfo=UTC),
        ),
        (
            datetime(2024, 2, 21, 0, 0, 0, tzinfo=UTC),
            datetime(2024, 3, 1, 0, 0, 0, tzinfo=UTC),
        ),
    ]
    data = {
        "start": "2024-01-01",
        "end": "2025-01-01",
        "cadence": {"type": "fixed", "interval": "dekadal"},
    }
    t = Temporal(**data)

    assert len(list(t.iter_date_ranges())) == 36

    assert list(t.iter_date_ranges())[:3] == [
        (
            datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 11, 0, 0, 0, tzinfo=UTC),
        ),
        (
            datetime(2024, 1, 11, 0, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 21, 0, 0, 0, tzinfo=UTC),
        ),
        (
            datetime(2024, 1, 21, 0, 0, 0, tzinfo=UTC),
            datetime(2024, 2, 1, 0, 0, 0, tzinfo=UTC),
        ),
    ]

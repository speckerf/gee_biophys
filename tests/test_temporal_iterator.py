from datetime import UTC, datetime


from gee_biophys.config import Temporal


def test_intervals():
    """Test correct interval generation for various cadences / fixed, seasonal."""

    # test valid monthly and bimonthly cadences
    data = {
        "start": "2023-01-01",
        "end": "2023-12-31",
        "cadence": {"type": "fixed", "interval": "monthly"},
    }
    t = Temporal(**data)

    assert t.start == datetime(2023, 1, 1, 0, 0, 0, tzinfo=UTC)
    assert t.end == datetime(2023, 12, 31, 0, 0, 0, tzinfo=UTC)

    assert len(list(t.iter_date_ranges())) == 12

    data = {
        "start": "2023-01-15",
        "end": "2023-04-14",
        "cadence": {"type": "fixed", "interval": "bimonthly"},
    }
    t = Temporal(**data)
    assert len(list(t.iter_date_ranges())) == 2

    # test valid seasonal cadence - in-year
    data = {
        "start": "2021-01-01",
        "end": "2023-12-31",
        "cadence": {"type": "seasons", "start": "05-15", "end": "09-15"},
    }
    t = Temporal(**data)
    assert len(list(t.iter_date_ranges())) == 3
    ranges = list(t.iter_date_ranges())
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

    # test valid seasonal cadence - cross-year
    data = {
        "start": "2020-01-01",
        "end": "2022-12-31",
        "cadence": {"type": "seasons", "start": "11-01", "end": "03-01"},
    }
    # this should raise warning, because start is within season
    t = Temporal(**data)
    assert len(list(t.iter_date_ranges())) == 4
    ranges = list(t.iter_date_ranges())
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

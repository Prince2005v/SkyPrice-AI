"""
Tests for src/preprocessing.py

Run with:
    pytest tests/test_preprocess.py -v
"""
import pytest
from datetime import datetime, date
from src.preprocessing import (
    preprocess_input,
    get_preprocessing_summary,
    VALID_AIRLINES,
    VALID_SOURCES,
    VALID_DESTINATIONS,
    FEATURE_ORDER,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_args():
    return dict(
        airline="IndiGo",
        source="Banglore",
        destination="New Delhi",
        journey_date=date(2025, 6, 15),
        dep_time=datetime(2025, 6, 15, 22, 20).time(),
        flight_class=0,
    )


# ─── Shape & Column Tests ──────────────────────────────────────────────────────

def test_output_shape(base_args):
    """Preprocessing must return exactly 1 row and 25 columns."""
    df = preprocess_input(**base_args)
    assert df.shape == (1, 25), f"Expected (1, 25), got {df.shape}"


def test_column_order(base_args):
    """Columns must exactly match FEATURE_ORDER."""
    df = preprocess_input(**base_args)
    assert list(df.columns) == FEATURE_ORDER


def test_all_integer_values(base_args):
    """All values must be integers (binary flags + numeric)."""
    df = preprocess_input(**base_args)
    assert df.dtypes.apply(lambda dt: dt == int).all(), "Not all columns are int dtype"


def test_ohe_values_binary(base_args):
    """One-hot encoded columns must only contain 0 or 1."""
    df = preprocess_input(**base_args)
    ohe_cols = [c for c in FEATURE_ORDER if c.startswith(("Airline_", "Source_", "Destination_"))]
    for col in ohe_cols:
        assert df[col].iloc[0] in (0, 1), f"Column {col} has non-binary value: {df[col].iloc[0]}"


# ─── Reference Category Tests (drop_first=True) ────────────────────────────────

def test_reference_airline_air_asia(base_args):
    """When airline is Air Asia (reference), all Airline_ flags should be 0."""
    args = {**base_args, "airline": "Air Asia"}
    df = preprocess_input(**args)
    airline_cols = [c for c in df.columns if c.startswith("Airline_")]
    assert df[airline_cols].sum(axis=1).iloc[0] == 0


def test_reference_source_banglore(base_args):
    """When source is Banglore (reference), all Source_ flags should be 0."""
    df = preprocess_input(**base_args)  # Banglore is already the source
    source_cols = [c for c in df.columns if c.startswith("Source_")]
    assert df[source_cols].sum(axis=1).iloc[0] == 0


def test_reference_destination_banglore():
    """Banglore as destination should encode all Destination_ flags as 0."""
    df = preprocess_input(
        airline="IndiGo",
        source="Mumbai",
        destination="Banglore",
        journey_date=date(2025, 6, 15),
        dep_time=datetime(2025, 6, 15, 10, 0).time(),
    )
    dest_cols = [c for c in df.columns if c.startswith("Destination_")]
    assert df[dest_cols].sum(axis=1).iloc[0] == 0


# ─── Feature Value Tests ───────────────────────────────────────────────────────

def test_correct_airline_flag(base_args):
    """IndiGo flag should be 1, all others 0."""
    df = preprocess_input(**base_args)
    assert df["Airline_IndiGo"].iloc[0] == 1
    other_airlines = [c for c in df.columns if c.startswith("Airline_") and c != "Airline_IndiGo"]
    assert df[other_airlines].sum(axis=1).iloc[0] == 0


def test_correct_datetime_extraction(base_args):
    """Journey day, month, dep hour, dep minute should be correctly extracted."""
    df = preprocess_input(**base_args)
    assert df["Journey_Day"].iloc[0] == 15
    assert df["Journey_Month"].iloc[0] == 6
    assert df["Dep_Hour"].iloc[0] == 22
    assert df["Dep_Minute"].iloc[0] == 20


def test_class_economy(base_args):
    df = preprocess_input(**base_args)
    assert df["Class"].iloc[0] == 0


def test_class_business():
    df = preprocess_input(
        airline="Jet Airways Business",
        source="Delhi",
        destination="Cochin",
        journey_date=date(2025, 8, 1),
        dep_time=datetime(2025, 8, 1, 9, 0).time(),
        flight_class=1,
    )
    assert df["Class"].iloc[0] == 1


def test_datetime_object_as_journey_date():
    """preprocess_input should accept datetime objects, not just date objects."""
    df = preprocess_input(
        airline="SpiceJet",
        source="Mumbai",
        destination="Delhi",
        journey_date=datetime(2025, 9, 10, 0, 0),  # datetime, not date
        dep_time=datetime(2025, 9, 10, 15, 30).time(),
    )
    assert df["Journey_Day"].iloc[0] == 10
    assert df["Journey_Month"].iloc[0] == 9


# ─── Validation / Error Tests ──────────────────────────────────────────────────

def test_invalid_airline_raises():
    with pytest.raises(ValueError, match="Invalid airline"):
        preprocess_input(
            airline="FakeAir",
            source="Mumbai",
            destination="Delhi",
            journey_date=date(2025, 6, 1),
            dep_time=datetime(2025, 6, 1, 10, 0).time(),
        )


def test_invalid_source_raises():
    with pytest.raises(ValueError, match="Invalid source"):
        preprocess_input(
            airline="IndiGo",
            source="Pune",
            destination="Delhi",
            journey_date=date(2025, 6, 1),
            dep_time=datetime(2025, 6, 1, 10, 0).time(),
        )


def test_invalid_destination_raises():
    with pytest.raises(ValueError, match="Invalid destination"):
        preprocess_input(
            airline="IndiGo",
            source="Mumbai",
            destination="Goa",
            journey_date=date(2025, 6, 1),
            dep_time=datetime(2025, 6, 1, 10, 0).time(),
        )


def test_same_source_destination_raises():
    with pytest.raises(ValueError, match="cannot be the same"):
        preprocess_input(
            airline="IndiGo",
            source="Delhi",
            destination="Delhi",
            journey_date=date(2025, 6, 1),
            dep_time=datetime(2025, 6, 1, 10, 0).time(),
        )


def test_invalid_flight_class_raises():
    with pytest.raises(ValueError, match="flight_class"):
        preprocess_input(
            airline="IndiGo",
            source="Mumbai",
            destination="Delhi",
            journey_date=date(2025, 6, 1),
            dep_time=datetime(2025, 6, 1, 10, 0).time(),
            flight_class=2,
        )


# ─── Summary Helper ─────────────────────────────────────────────────────────────

def test_summary_keys(base_args):
    summary = get_preprocessing_summary(**base_args)
    expected_keys = {"route", "airline", "travel_class", "journey_date", "dep_time", "feature_shape", "active_flags"}
    assert expected_keys.issubset(summary.keys())


def test_summary_route_format(base_args):
    summary = get_preprocessing_summary(**base_args)
    assert "→" in summary["route"]


# ─── Coverage Across All Airlines ─────────────────────────────────────────────

@pytest.mark.parametrize("airline", VALID_AIRLINES)
def test_all_airlines_encode_without_error(airline):
    df = preprocess_input(
        airline=airline,
        source="Mumbai",
        destination="Delhi",
        journey_date=date(2025, 7, 20),
        dep_time=datetime(2025, 7, 20, 8, 0).time(),
    )
    assert df.shape == (1, 25)


@pytest.mark.parametrize("src", VALID_SOURCES)
def test_all_sources_encode_without_error(src):
    # Avoid src == dest collision
    dest = "Delhi" if src != "Delhi" else "Cochin"
    df = preprocess_input(
        airline="IndiGo",
        source=src,
        destination=dest,
        journey_date=date(2025, 7, 20),
        dep_time=datetime(2025, 7, 20, 8, 0).time(),
    )
    assert df.shape == (1, 25)

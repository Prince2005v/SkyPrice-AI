import pandas as pd
import numpy as np
from datetime import date, datetime
import time as time_module

# ─── Constants ────────────────────────────────────────────────────────────────

VALID_AIRLINES = [
    'Air Asia', 'Air India', 'GoAir', 'IndiGo', 'Jet Airways',
    'Jet Airways Business', 'Multiple carriers',
    'Multiple carriers Premium economy', 'SpiceJet',
    'Trujet', 'Vistara', 'Vistara Premium economy'
]

VALID_SOURCES = ['Banglore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai']
VALID_DESTINATIONS = ['Banglore', 'Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi']

# OHE reference sets (drop_first=True: Air Asia & Banglore are reference categories)
_AIRLINE_OHE = [a for a in VALID_AIRLINES if a != 'Air Asia']
_SOURCE_OHE   = [s for s in VALID_SOURCES if s != 'Banglore']
_DEST_OHE     = [d for d in VALID_DESTINATIONS if d != 'Banglore']

FEATURE_ORDER = [
    'Class', 'Journey_Day', 'Journey_Month', 'Dep_Hour', 'Dep_Minute',
    'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
    'Airline_Jet Airways Business', 'Airline_Multiple carriers',
    'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
    'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
    'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
    'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
    'Destination_Kolkata', 'Destination_New Delhi'
]


# ─── Validation helpers ────────────────────────────────────────────────────────

def _validate_inputs(airline, source, destination, journey_date, dep_time, flight_class):
    """Raise ValueError with a clear message if any input is invalid."""
    if airline not in VALID_AIRLINES:
        raise ValueError(
            f"Invalid airline '{airline}'. "
            f"Must be one of: {', '.join(VALID_AIRLINES)}"
        )
    if source not in VALID_SOURCES:
        raise ValueError(
            f"Invalid source city '{source}'. "
            f"Must be one of: {', '.join(VALID_SOURCES)}"
        )
    if destination not in VALID_DESTINATIONS:
        raise ValueError(
            f"Invalid destination city '{destination}'. "
            f"Must be one of: {', '.join(VALID_DESTINATIONS)}"
        )
    if source == destination:
        raise ValueError("Source and destination cities cannot be the same.")
    if not isinstance(flight_class, int) or flight_class not in (0, 1):
        raise ValueError(
            f"Invalid flight_class '{flight_class}'. Must be 0 (Economy) or 1 (Business)."
        )


# ─── Public API ───────────────────────────────────────────────────────────────

def preprocess_input(airline, source, destination, journey_date, dep_time, flight_class=0):
    """
    Preprocess raw user inputs into a feature DataFrame ready for model inference.

    Parameters
    ----------
    airline : str
        Airline name. Must be one of VALID_AIRLINES.
    source : str
        Origin city. Must be one of VALID_SOURCES.
    destination : str
        Destination city. Must be one of VALID_DESTINATIONS.
    journey_date : datetime.date | datetime
        Date of travel.
    dep_time : datetime.time
        Departure time.
    flight_class : int, optional
        0 = Economy, 1 = Business/Premium. Default 0.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with 25 binary/integer features matching
        the training feature set, column order guaranteed.

    Raises
    ------
    ValueError
        If any input fails validation.
    """
    _validate_inputs(airline, source, destination, journey_date, dep_time, flight_class)

    # Normalise date/time: accept both date and datetime objects
    if isinstance(journey_date, datetime):
        journey_date = journey_date.date()

    data = {
        'Class':          flight_class,
        'Journey_Day':    journey_date.day,
        'Journey_Month':  journey_date.month,
        'Dep_Hour':       dep_time.hour,
        'Dep_Minute':     dep_time.minute,
    }

    # Airline OHE  (reference: Air Asia → all zeros)
    for a in _AIRLINE_OHE:
        data[f'Airline_{a}'] = int(airline == a)

    # Source OHE   (reference: Banglore → all zeros)
    for s in _SOURCE_OHE:
        data[f'Source_{s}'] = int(source == s)

    # Destination OHE  (reference: Banglore → all zeros)
    for d in _DEST_OHE:
        data[f'Destination_{d}'] = int(destination == d)

    df = pd.DataFrame([data])
    return df[FEATURE_ORDER].astype(int)


def get_preprocessing_summary(airline, source, destination, journey_date, dep_time, flight_class=0):
    """
    Return a human-readable summary dict of the encoded features.
    Useful for debugging and display in the UI.
    """
    df = preprocess_input(airline, source, destination, journey_date, dep_time, flight_class)
    return {
        "route":         f"{source}  →  {destination}",
        "airline":       airline,
        "travel_class":  "Business / Premium" if flight_class else "Economy",
        "journey_date":  str(journey_date),
        "dep_time":      f"{dep_time.hour:02d}:{dep_time.minute:02d}",
        "feature_shape": df.shape,
        "active_flags":  int(df.sum(axis=1).iloc[0]),
    }

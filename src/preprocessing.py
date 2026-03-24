import pandas as pd
import numpy as np

def preprocess_input(airline, source, destination, journey_date, dep_time, flight_class=0):
    """
    Preprocess inputs to match the model's expected features.
    
    Parameters:
    - airline (str): Name of the airline.
    - source (str): Source city.
    - destination (str): Destination city.
    - journey_date (datetime): Date of journey.
    - dep_time (datetime.time): Departure time.
    - flight_class (int): Class of flight (default is 0).
    
    Returns:
    - pd.DataFrame: Processed features ready for model prediction.
    """
    
    # Initialize dictionary with features from feature_columns.pkl
    data = {
        'Class': flight_class,
        'Journey_Day': journey_date.day,
        'Journey_Month': journey_date.month,
        'Dep_Hour': dep_time.hour,
        'Dep_Minute': dep_time.minute
    }
    
    # Airline One-Hot Encoding (drop_first=True, Air Asia dropped)
    airlines = [
        'Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business',
        'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet',
        'Trujet', 'Vistara', 'Vistara Premium economy'
    ]
    for a in airlines:
        data[f'Airline_{a}'] = 1 if airline == a else 0
        
    # Source One-Hot Encoding (drop_first=True, Banglore dropped)
    sources = ['Chennai', 'Delhi', 'Kolkata', 'Mumbai']
    for s in sources:
        data[f'Source_{s}'] = 1 if source == s else 0
        
    # Destination One-Hot Encoding (drop_first=True, Banglore dropped)
    destinations = ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi']
    for d in destinations:
        data[f'Destination_{d}'] = 1 if destination == d else 0
        
    # Create DataFrame and ensure column order matches feature_columns.pkl
    df = pd.DataFrame([data])
    
    feature_order = [
        'Class', 'Journey_Day', 'Journey_Month', 'Dep_Hour', 'Dep_Minute',
        'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
        'Airline_Jet Airways Business', 'Airline_Multiple carriers',
        'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
        'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
        'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
        'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
        'Destination_Kolkata', 'Destination_New Delhi'
    ]
    
    return df[feature_order].astype(int)

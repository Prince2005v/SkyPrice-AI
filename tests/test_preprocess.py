from src.preprocessing import preprocess_input
from datetime import datetime
import pandas as pd

def test_preprocessing():
    # Mock inputs
    airline = 'IndiGo'
    source = 'Banglore'
    destination = 'New Delhi'
    journey_date = datetime(2019, 3, 24)
    dep_time = datetime(2019, 3, 24, 22, 20).time()
    
    # Process
    try:
        features = preprocess_input(airline, source, destination, journey_date, dep_time)
        print("Features Shape:", features.shape)
        print("Features Columns:", list(features.columns))
        print("First Row Values:", features.iloc[0].values)
        
        expected_cols = 25
        if features.shape[1] == expected_cols:
            print("SUCCESS: Feature count matches expectations (25).")
        else:
            print(f"FAILURE: Expected {expected_cols} columns, got {features.shape[1]}")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_preprocessing()

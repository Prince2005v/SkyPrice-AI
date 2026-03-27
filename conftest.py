"""
Root conftest.py — ensures the project root is on sys.path so that
`from src.preprocessing import ...` works when pytest is run from any directory.
"""
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(__file__))

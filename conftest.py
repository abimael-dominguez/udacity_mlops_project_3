import pytest
import pandas as pd


@pytest.fixture(scope="session")
def data():
    """Provides the data for the tests"""
    local_path = "./data/clean_census.csv"
    df = pd.read_csv(local_path, low_memory=False)
    return df

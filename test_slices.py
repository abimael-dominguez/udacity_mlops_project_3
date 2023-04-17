"""
Tests to evaluate the performance of each category
in categorical features.

Note: The data is used is loaded in the conftest.py
using fixtures.

To run the this test:
    pytest -vv test_slices.py
"""

from starter.ml.model import inference

def test_data_shape(data):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert data.shape == data.dropna().shape, "Dropping null changes shape."


def test_slice_hours_per_week_by_occupation(data):
    """ Test to see if our mean per categorical slice is in the range 5 to 168.
        Or in other words, I week have 168 hours so a number greater than that can't happen,
        and at leas 5 hours per week are allowed. All this regardless the occupation.
    """
    for cat_feat in data["occupation"].unique():
        avg_value = data[data["occupation"] == cat_feat]["hours-per-week"].mean()
        assert (
            5 < avg_value < 168
        ), f"For {cat_feat}, average of {avg_value} not between 5 and 168 hours."


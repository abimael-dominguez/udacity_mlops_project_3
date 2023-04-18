"""
Tests to evaluate the performance of each category
in categorical features.

Note: The data is used is loaded in the conftest.py
using fixtures.

To run the this test:
    pytest -vv test_slices.py
"""


def test_data_shape(data):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert data.shape == data.dropna().shape, "Dropping null changes shape."


def test_slice_hours_per_week_by_occupation(data):
    """ Test to see if our mean per categorical slice is in the range 5 to 168.
        Or in other words, I week have 168 hours so a number greater than that can't happen,
        and at leas 5 hours per week are allowed. All this regardless the occupation.
    """
    for cat_feat in data["occupation"].unique():
        avg_value = data[data["occupation"] ==
                         cat_feat]["hours-per-week"].mean()
        assert (
            5 < avg_value < 168
        ), f"For {cat_feat}, average of {avg_value} not between 5 and 168 hours."


def test_precision_united_states(data_slices_performance):
    """
    Check wether the precision for {"native-country ": "United-States"}
    is acceptable (i.e precision >= 0.5)

    From last training results:
        feature,category,precision,recall,fbeta
        native-country,United-States,0.757,0.3028,0.4326
    """
    mask_1 = data_slices_performance["feature"] == "native-country"
    mask_2 = data_slices_performance["category"] == "United-States"
    data = data_slices_performance[mask_1 & mask_2]

    assert data['precision'].max() > 0.5


def test_overall_avg_recall(data_slices_performance):
    """
    Using the slice_output.csv calculates 
    the average of the recall column.
    """
    assert data_slices_performance['recall'].mean() > 0.1


def test_overall_avg_fbeta(data_slices_performance):
    """
    Using the slice_output.csv calculates 
    the average of the fbeta column.
    """
    assert data_slices_performance['fbeta'].mean() > 0.3
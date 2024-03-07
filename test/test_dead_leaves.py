import numpy as np
from rstor.synthetic_data.dead_leaves import dead_leaves_chart


def test_dead_leaves_chart():
    # Test case 1: Default parameters
    chart = dead_leaves_chart()
    assert isinstance(chart, np.ndarray)
    assert chart.shape == (100, 100, 3)

    # Test case 2: Custom size and number of circles
    chart = dead_leaves_chart(size=(200, 150), number_of_circles=10)
    assert isinstance(chart, np.ndarray)
    assert chart.shape == (200, 150, 3)

    # Test case 3: Colored circles
    chart = dead_leaves_chart(colored=True)
    assert isinstance(chart, np.ndarray)
    assert chart.shape == (100, 100, 3)

    # Test case 4: Custom radius mean and stddev
    chart = dead_leaves_chart(radius_min=5, radius_alpha=2)
    assert isinstance(chart, np.ndarray)
    assert chart.shape == (100, 100, 3)

    # Test case 5: Custom background color
    chart = dead_leaves_chart(background_color=(0.2, 0.4, 0.6))
    assert isinstance(chart, np.ndarray)
    assert chart.shape == (100, 100, 3)

    # Test case 6: Custom seed
    chart1 = dead_leaves_chart(seed=42)
    chart2 = dead_leaves_chart(seed=42)
    assert np.array_equal(chart1, chart2)

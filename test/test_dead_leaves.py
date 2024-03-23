import numpy as np
from rstor.synthetic_data.dead_leaves_cpu import cpu_dead_leaves_chart
from rstor.properties import SAMPLER_NATURAL, DATASET_PATH


def test_dead_leaves_chart():
    # Test case 1: Default parameters
    chart = cpu_dead_leaves_chart()
    assert isinstance(chart, np.ndarray)
    assert chart.shape == (100, 100, 3)

    # Test case 2: Custom size and number of circles
    chart = cpu_dead_leaves_chart(size=(200, 150), number_of_circles=10)
    assert isinstance(chart, np.ndarray)
    assert chart.shape == (200, 150, 3)

    # Test case 3: Colored circles
    chart = cpu_dead_leaves_chart(colored=True, number_of_circles=300)
    assert isinstance(chart, np.ndarray)
    assert chart.shape == (100, 100, 3)

    # Test case 4: Custom radius mean and stddev
    chart = cpu_dead_leaves_chart(radius_min=5, radius_alpha=2, number_of_circles=300)
    assert isinstance(chart, np.ndarray)
    assert chart.shape == (100, 100, 3)

    # Test case 5: Custom background color
    chart = cpu_dead_leaves_chart(background_color=(0.2, 0.4, 0.6), number_of_circles=300)
    assert isinstance(chart, np.ndarray)
    assert chart.shape == (100, 100, 3)

    # Test case 6: Custom seed
    chart1 = cpu_dead_leaves_chart(seed=42, number_of_circles=300)
    chart2 = cpu_dead_leaves_chart(seed=42, number_of_circles=300)
    assert np.array_equal(chart1, chart2)


def test_dead_leaves_color_sampler():
    img_list = sorted(
        list((DATASET_PATH / "sample").glob("*.png"))
    )
    _gen = cpu_dead_leaves_chart(number_of_circles=300, sampler=SAMPLER_NATURAL, natural_image_list=img_list)
    # from interactive_pipe.data_objects.image import Image
    # Image(_gen).show()

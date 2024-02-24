from rstor.synthetic_data.dead_leaves import dead_leaves_chart
import sys
import numpy as np
from interactive_pipe import interactive_pipeline, interactive
from typing import Optional


@interactive(
    background_intensity=(0., 1., 0.5),
    number_of_circles=(-1, 10000, -1),
    colored=(False,),
    radius_mean=(-1., 200., -1),
    radius_stddev=(-1., 100., -1)
)
def generate_deadleave(
    background_intensity: float = 0.5,
    number_of_circles: int = -1,
    colored: Optional[bool] = False,
    radius_mean: Optional[int] = -1,
    radius_stddev: Optional[int] = -1,
) -> np.ndarray:
    bg_color = (background_intensity, background_intensity, background_intensity)
    chart = dead_leaves_chart((600, 600), number_of_circles, bg_color, colored, radius_mean, radius_stddev)
    return chart


def deadleave_pipeline():
    deadleave_chart = generate_deadleave()
    return deadleave_chart


def main(argv):
    interactive_pipeline(gui="auto")(deadleave_pipeline)()


if __name__ == "__main__":
    main(sys.argv[1:])

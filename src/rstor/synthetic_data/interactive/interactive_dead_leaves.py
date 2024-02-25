from rstor.synthetic_data.dead_leaves import dead_leaves_chart
import sys
import numpy as np
from interactive_pipe import interactive_pipeline, interactive
from typing import Optional


@interactive(
    background_intensity=(0.5, [0., 1.]),
    number_of_circles=(-1, [-1, 10000]),
    colored=(False,),
    radius_mean=(-1., [-1., 200]),
    radius_stddev=(-1., [-1., 100.]),
    seed=(0, [-1, 42])
)
def generate_deadleave(
    background_intensity: float = 0.5,
    number_of_circles: int = -1,
    colored: Optional[bool] = False,
    radius_mean: Optional[int] = -1,
    radius_stddev: Optional[int] = -1,
    seed=0
) -> np.ndarray:
    bg_color = (background_intensity, background_intensity, background_intensity)
    chart = dead_leaves_chart((200, 200), number_of_circles, bg_color, colored, radius_mean, radius_stddev,
                              seed=None if seed < 0 else seed)
    return chart


def deadleave_pipeline():
    deadleave_chart = generate_deadleave()
    return deadleave_chart


def main(argv):
    interactive_pipeline(gui="auto")(deadleave_pipeline)()


if __name__ == "__main__":
    main(sys.argv[1:])

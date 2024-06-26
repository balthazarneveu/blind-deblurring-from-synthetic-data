from rstor.synthetic_data.dead_leaves_cpu import cpu_dead_leaves_chart
from rstor.synthetic_data.dead_leaves_gpu import gpu_dead_leaves_chart
from rstor.properties import SAMPLER_UNIFORM, SAMPLER_DIV2K, SAMPLER_NATURAL, SAMPLER_SATURATED, DATASET_PATH
import sys
import numpy as np
from interactive_pipe import interactive_pipeline, interactive
from typing import Optional


def dead_leave_plugin(ds=1):
    interactive(
        background_intensity=(0.5, [0., 1.]),
        number_of_circles=(-1, [-1, 10000]),
        colored=(True,),
        radius_alpha=(3., [2., 10.]),
        seed=(0, [-1, 42]),
        ds=(ds, [1, 5]),
        numba_flag=(True,),  # Default CPU to avoid issues by default
        sampler=(SAMPLER_UNIFORM, [SAMPLER_UNIFORM, SAMPLER_DIV2K, SAMPLER_SATURATED]),
        circle_primitives=(True,),
        anisotropy=(1., [0.1, 10.]),
        angle=(0., [-180., 180.])
        # ds=(ds, [1, 5])
    )(generate_deadleave)


def generate_deadleave(
    background_intensity: float = 0.5,
    number_of_circles: int = -1,
    colored: Optional[bool] = False,
    radius_alpha: Optional[int] = 3,
    seed=0,
    ds=3,
    numba_flag=True,
    sampler=SAMPLER_UNIFORM,
    circle_primitives=True,
    anisotropy=1.,
    angle=0.,
    global_params={}
) -> np.ndarray:
    global_params["ds_factor"] = ds
    bg_color = (background_intensity, background_intensity, background_intensity)
    natural_image_list = None
    if sampler == SAMPLER_DIV2K:
        sampler = SAMPLER_NATURAL
        div2k_path = DATASET_PATH / "div2k" / "DIV2K_train_HR" / "DIV2K_train_HR"
        natural_image_list = sorted([file for file in div2k_path.glob("*.png")])
    if not numba_flag:
        chart = cpu_dead_leaves_chart((512*ds, 512*ds), number_of_circles, bg_color, colored,
                                      radius_alpha=radius_alpha,
                                      seed=None if seed < 0 else seed,
                                      sampler=sampler,
                                      reverse=False,
                                      natural_image_list=natural_image_list)
    else:
        chart = gpu_dead_leaves_chart((512*ds, 512*ds), number_of_circles, bg_color, colored,
                                      radius_alpha=radius_alpha,
                                      seed=None if seed < 0 else seed,
                                      sampler=sampler,
                                      natural_image_list=natural_image_list,
                                      circle_primitives=circle_primitives,
                                      anisotropy=anisotropy,
                                      angle=angle).copy_to_host()
    if chart.shape[-1] == 1:
        chart = chart.repeat(3, axis=-1)
        # Required to switch from colors to gray scale visualization.
    return chart


def deadleave_pipeline():
    deadleave_chart = generate_deadleave()
    return deadleave_chart


def main(argv):
    dead_leave_plugin(ds=1)
    interactive_pipeline(gui="auto")(deadleave_pipeline)()


if __name__ == "__main__":
    main(sys.argv[1:])

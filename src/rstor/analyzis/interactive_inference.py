from rstor.synthetic_data.interactive.interactive_dead_leaves import generate_deadleave
from rstor.analyzis.interactive.crop import crop_selector, crop
import sys
from interactive_pipe import interactive_pipeline
from rstor.analyzis.interactive.inference import infer
from rstor.analyzis.interactive.degradation import degrade
from rstor.analyzis.interactive.model_selection import model_selector, get_default_models


def deadleave_inference_pipeline(models_dict: dict):
    groundtruth = generate_deadleave()
    model = model_selector(models_dict)
    degraded_chart = degrade(groundtruth)
    restored_chart = infer(degraded_chart, model)
    crop_selector(restored_chart)
    groundtruth, degraded_chart, restored_chart = crop(groundtruth, degraded_chart, restored_chart)
    return groundtruth, degraded_chart, restored_chart


def main(argv):
    model_dict = get_default_models()
    interactive_pipeline(gui="auto", cache=True, safe_input_buffer_deepcopy=False)(
        deadleave_inference_pipeline)(model_dict)


if __name__ == "__main__":
    main(sys.argv[1:])

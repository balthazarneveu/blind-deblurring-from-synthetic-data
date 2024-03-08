import sys
from interactive_pipe import interactive_pipeline
from rstor.synthetic_data.interactive.interactive_dead_leaves import dead_leave_plugin
from rstor.analyzis.interactive.pipelines import deadleave_inference_pipeline
from rstor.analyzis.interactive.model_selection import get_default_models
from rstor.analyzis.interactive.crop import plug_crop_selector
from pathlib import Path
from rstor.analyzis.parser import get_parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    plug_crop_selector(num_pad=args.keyboard)
    model_dict = get_default_models(args.experiments, Path(args.models_storage))
    dead_leave_plugin(ds=1)
    interactive_pipeline(gui="auto", cache=True, safe_input_buffer_deepcopy=False)(
        deadleave_inference_pipeline)(model_dict)


if __name__ == "__main__":
    main(sys.argv[1:])

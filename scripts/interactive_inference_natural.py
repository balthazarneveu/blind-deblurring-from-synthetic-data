import sys
from interactive_pipe import interactive_pipeline
from rstor.analyzis.interactive.pipelines import natural_inference_pipeline
from rstor.analyzis.interactive.model_selection import get_default_models
from pathlib import Path
from rstor.analyzis.parser import get_parser
import argparse
from batch_processing import Batch
from interactive_pipe.data_objects.image import Image
from rstor.analyzis.interactive.images import image_selector
from rstor.analyzis.interactive.crop import plug_crop_selector
from rstor.analyzis.interactive.metrics import plug_configure_metrics
from interactive_pipe import interactive, KeyboardControl


def image_loading_batch(input: Path, args: argparse.Namespace) -> dict:
    """Wrapper to load images files from a directory using batch_processing
    """

    if not args.disable_preload:
        img = Image.from_file(input).data
        return {"name": input.name, "path": input, "buffer": img}
    else:
        return {"name": input.name, "path": input, "buffer": None}


def main(argv):
    batch = Batch(argv)
    batch.set_io_description(
        input_help='input image files',
        output_help=argparse.SUPPRESS
    )
    parser = get_parser()
    parser.add_argument("-nop", "--disable-preload", action="store_true", help="Disable images preload")
    args = batch.parse_args(parser)
    # batch.set_multiprocessing_enabled(False)
    img_list = batch.run(image_loading_batch)
    if args.keyboard:
        image_control = KeyboardControl(0, [0, len(img_list)-1], keydown="3", keyup="9", modulo=True)
    else:
        image_control = (0, [0, len(img_list)-1])
    interactive(image_index=image_control)(image_selector)
    plug_crop_selector(num_pad=args.keyboard)
    plug_configure_metrics(key_shortcut="a")  # "a" if args.keyboard else None)
    model_dict = get_default_models(args.experiments, Path(args.models_storage), keyboard_control=args.keyboard)
    interactive_pipeline(
        gui="auto",
        cache=True,
        safe_input_buffer_deepcopy=False
    )(natural_inference_pipeline)(
        img_list,
        model_dict
    )


if __name__ == "__main__":
    main(sys.argv[1:])

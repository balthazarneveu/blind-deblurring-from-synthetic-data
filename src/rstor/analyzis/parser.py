from rstor.analyzis.interactive.model_selection import MODELS_PATH
import argparse


def get_models_parser(parser: argparse.ArgumentParser = None, help: str = "Inference",
                      default_models_path: str = MODELS_PATH) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description=help)
    parser.add_argument("-e", "--experiments", type=int, nargs="+", required=True,
                        help="Experience indexes to be used at inference time")
    parser.add_argument("-m", "--models-storage", type=str, help="Model storage path", default=default_models_path)
    return parser


def get_parser(
    parser: argparse.ArgumentParser = None,
    help: str = "Live inference pipeline"
) -> argparse.ArgumentParser:
    """Generic parser for live interactive inference
    """
    if parser is None:
        parser = argparse.ArgumentParser(description=help)
    get_models_parser(parser=parser, help=help)
    parser.add_argument("-k", "--keyboard", action="store_true", help="Keyboard control - less sliders")
    return parser

from rstor.analyzis.interactive.model_selection import MODELS_PATH
import argparse


def get_parser(parser: argparse.ArgumentParser = None, help: str = "Live inference pipeline"):
    if parser is None:
        parser = argparse.ArgumentParser(description=help)
    parser.add_argument("-e", "--experiments", type=int, nargs="+", required=True,
                        help="Experience indexes to be used at inference time")
    parser.add_argument("-m", "--models-storage", type=str, help="Model storage path", default=MODELS_PATH)
    return parser

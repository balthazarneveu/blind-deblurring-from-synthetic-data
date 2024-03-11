from configuration import ROOT_DIR, OUTPUT_FOLDER_NAME, INFERENCE_FOLDER_NAME
from rstor.analyzis.parser import get_models_parser
from batch_processing import Batch
from rstor.properties import (
    DEVICE, NAME, PRETTY_NAME, DATALOADER, CONFIG_DEAD_LEAVES, VALIDATION,
    BATCH_SIZE, SIZE,
    REDUCTION_AVERAGE, REDUCTION_SKIP,
    TRACES_TARGET, TRACES_DEGRADED, TRACES_RESTORED, TRACES_METRICS, TRACES_ALL
)
from rstor.data.dataloader import get_data_loader
from tqdm import tqdm
from pathlib import Path
import torch
from typing import Optional
import argparse
import sys
from rstor.analyzis.interactive.model_selection import get_default_models
from rstor.learning.metrics import compute_metrics
from interactive_pipe.data_objects.image import Image
from interactive_pipe.data_objects.parameters import Parameters
from typing import List
from itertools import product
import pandas as pd
ALL_TRACES = [TRACES_TARGET, TRACES_DEGRADED, TRACES_RESTORED, TRACES_METRICS]


def parse_int_pairs(s):
    try:
        # Split the input string by spaces to separate pairs, then split each pair by ',' and convert to tuple of ints
        return [tuple(map(int, item.split(','))) for item in s.split()]
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a series of pairs 'a,b' separated by spaces.")


def get_parser(parser: Optional[argparse.ArgumentParser] = None, batch_mode=False) -> argparse.ArgumentParser:
    parser = get_models_parser(
        parser=parser,
        help="Inference on validation set",
        default_models_path=ROOT_DIR/OUTPUT_FOLDER_NAME)
    if not batch_mode:
        parser.add_argument("-o", "--output-dir", type=str, default=ROOT_DIR /
                            INFERENCE_FOLDER_NAME, help="Output directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--traces", "-t", nargs="+", type=str, choices=ALL_TRACES+[TRACES_ALL],
                        help="Traces to be computed", default=TRACES_ALL)
    parser.add_argument("--size", type=parse_int_pairs,
                        default=[(256, 256)], help="Size of the images like '256,512 512,512'")
    parser.add_argument("--std-dev", type=parse_int_pairs, default=[(0, 50)],
                        help="Noise standard deviation (a, b) as pairs separated by spaces, e.g., '0,50 8,8 6,10'")
    parser.add_argument("-n", "--number-of-images", type=int, default=None,
                        required=False, help="Number of images to process")
    return parser


def to_image(img: torch.Tensor):
    return img.permute(0, 2, 3, 1).cpu().numpy()


def infer(model, dataloader, config, device, output_dir: Path, traces: List[str] = ALL_TRACES, number_of_images=None):
    img_index = 0
    if TRACES_ALL in traces:
        traces = ALL_TRACES
    if TRACES_METRICS in traces:
        all_metrics = {}
    else:
        all_metrics = None
    with torch.no_grad():
        model.eval()
        for img_degraded, img_target in tqdm(dataloader):
            img_degraded = img_degraded.to(device)
            img_target = img_target.to(device)
            img_restored = model(img_degraded)
            if TRACES_METRICS in traces:
                metrics_input_per_image = compute_metrics(img_degraded, img_target, reduction=REDUCTION_SKIP)
                metrics_per_image = compute_metrics(img_restored, img_target, reduction=REDUCTION_SKIP)
                # print(metrics_per_image)
            img_degraded = to_image(img_degraded)
            img_target = to_image(img_target)
            img_restored = to_image(img_restored)
            for idx in range(img_restored.shape[0]):
                degradation_parameters = dataloader.dataset.current_degradation[img_index]
                common_prefix = f"{img_index:05d}_{img_degraded.shape[-3]:04d}x{img_degraded.shape[-2]:04d}"
                common_prefix += f"_noise=[{config[DATALOADER][CONFIG_DEAD_LEAVES]['noise_stddev'][0]:02d},{config[DATALOADER][CONFIG_DEAD_LEAVES]['noise_stddev'][1]:02d}]"
                suffix_deg = f"_noise={round(degradation_parameters['noise_stddev']):02d}"
                save_path_pred = output_dir/f"{common_prefix}_pred{suffix_deg}_{config[PRETTY_NAME]}.png"
                save_path_degr = output_dir/f"{common_prefix}_degr{suffix_deg}.png"
                save_path_targ = output_dir/f"{common_prefix}_targ.png"
                if TRACES_RESTORED in traces:
                    Image(img_restored[idx]).save(save_path_pred)
                if TRACES_DEGRADED in traces:
                    Image(img_degraded[idx]).save(save_path_degr)
                if TRACES_TARGET in traces:
                    Image(img_target[idx]).save(save_path_targ)
                if TRACES_METRICS in traces:
                    # current_metrics = {"in": {}, "out": {}}
                    # for key, value in metrics_per_image.items():
                    #     print(f"{key}: {value[idx]:.3f}")
                    #     current_metrics["in"][key] = metrics_input_per_image[key][idx].item()
                    #     current_metrics["out"][key] = metrics_per_image[key][idx].item()
                    current_metrics = {}
                    for key, value in metrics_per_image.items():
                        current_metrics["in_"+key] = metrics_input_per_image[key][idx].item()
                        current_metrics["out_"+key] = metrics_per_image[key][idx].item()
                    current_metrics["degradation"] = degradation_parameters
                    current_metrics["size"] = (img_degraded.shape[-3], img_degraded.shape[-2])
                    current_metrics["deadleaves_config"] = config[DATALOADER][CONFIG_DEAD_LEAVES]
                    current_metrics["restored"] = save_path_pred.relative_to(output_dir).as_posix()
                    current_metrics["degraded"] = save_path_degr.relative_to(output_dir).as_posix()
                    current_metrics["target"] = save_path_targ.relative_to(output_dir).as_posix()
                    current_metrics["model"] = config[PRETTY_NAME]
                    current_metrics["model_id"] = config[NAME]
                    Parameters(current_metrics).save(output_dir/f"{common_prefix}_metrics.json")
                    # for key, value in all_metrics.items():

                    all_metrics[img_index] = current_metrics
                img_index += 1
                if number_of_images is not None and img_index > number_of_images:
                    return all_metrics
    return all_metrics


def infer_main(argv, batch_mode=False):
    parser = get_parser(batch_mode=batch_mode)
    if batch_mode:
        batch = Batch(argv)
        batch.set_io_description(
            input_help='input image files',
            output_help=f'output directory {str(ROOT_DIR/INFERENCE_FOLDER_NAME)}',
        )
        batch.parse_args(parser)
    else:
        args = parser.parse_args(argv)
    device = "cpu" if args.cpu else DEVICE

    for exp in args.experiments:
        model_dict = get_default_models([exp], Path(args.models_storage), interactive_flag=False)
        # print(list(model_dict.keys()))
        current_model_dict = model_dict[list(model_dict.keys())[0]]
        model = current_model_dict["model"]
        config = current_model_dict["config"]
        for std_dev, size in product(args.std_dev, args.size):
            config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(
                blur_kernel_half_size=[0, 0],
                ds_factor=1,
                noise_stddev=list(std_dev)
            )
            config[DATALOADER][SIZE] = size
            config[DATALOADER][BATCH_SIZE][VALIDATION] = 4
            dataloader = get_data_loader(config, frozen_seed=42)
            # print(config)
            output_dir = Path(args.output_dir)/(config[NAME] + "_" +
                                                config[PRETTY_NAME]) #+ "_" + f"{size[0]:04d}x{size[1]:04d}")
            output_dir.mkdir(parents=True, exist_ok=True)

            all_metrics = infer(model, dataloader[VALIDATION], config, device, output_dir,
                                traces=args.traces, number_of_images=args.number_of_images)
            if all_metrics is not None:
                # print(all_metrics)
                df = pd.DataFrame(all_metrics).T
                prefix = f"{size[0]:04d}x{size[1]:04d}_noise=[{std_dev[0]:02d},{std_dev[1]:02d}]" + \
                    f"_{config[PRETTY_NAME]}"
                df.to_csv(output_dir/f"__{prefix}_metrics_.csv", index=False)
                # Normally this could go into another script to handle the metrics analyzis
                # print(df)


if __name__ == "__main__":
    infer_main(sys.argv[1:])

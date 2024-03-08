from configuration import ROOT_DIR, OUTPUT_FOLDER_NAME, INFERENCE_FOLDER_NAME
from rstor.analyzis.parser import get_models_parser
from batch_processing import Batch
from rstor.properties import (
    DEVICE, NAME, PRETTY_NAME, DATALOADER, CONFIG_DEAD_LEAVES, VALIDATION,
    BATCH_SIZE, SIZE
)
from rstor.data.dataloader import get_data_loader
from tqdm import tqdm
from pathlib import Path
import torch
from typing import Optional
import argparse
import sys
from rstor.analyzis.interactive.model_selection import get_default_models
from interactive_pipe.data_objects.image import Image


def get_parser(parser: Optional[argparse.ArgumentParser] = None, batch_mode=False) -> argparse.ArgumentParser:
    parser = get_models_parser(
        parser=parser,
        help="Inference on validation set",
        default_models_path=ROOT_DIR/OUTPUT_FOLDER_NAME)
    if not batch_mode:
        parser.add_argument("-o", "--output-dir", type=str, default=ROOT_DIR /
                            INFERENCE_FOLDER_NAME, help="Output directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser


def to_image(img: torch.Tensor):
    return img.permute(0, 2, 3, 1).cpu().numpy()


def infer(model, dataloader, config, device, output_dir: Path):
    img_index = 0
    with torch.no_grad():
        model.eval()
        for img_degraded, img_target in tqdm(dataloader):
            # print(img_degraded.shape, img_target.shape)
            img_degraded = img_degraded.to(device)
            img_target = img_target.to(device)
            img_restored = model(img_degraded)
            # compute metrics here!!
            img_degraded = to_image(img_degraded)
            img_target = to_image(img_target)
            img_restored = to_image(img_restored)
            for idx in range(img_restored.shape[0]):
                degradation_parameters = dataloader.dataset.current_degradation[img_index]
                common_prefix = f"{img_index:05d}_{img_degraded.shape[-3]}x{img_degraded.shape[-2]}"
                suffix_deg = f"_noise={degradation_parameters['noise_stddev']:.1f}"
                save_path_pred = output_dir/f"{common_prefix}_pred{suffix_deg}_{config[PRETTY_NAME]}.png"
                save_path_degr = output_dir/f"{common_prefix}_degr{suffix_deg}.png"
                save_path_targ = output_dir/f"{common_prefix}_targ.png"
                Image(img_restored[idx]).save(save_path_pred)
                Image(img_degraded[idx]).save(save_path_degr)
                Image(img_target[idx]).save(save_path_targ)
                img_index += 1
                if img_index > 10:
                    return


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
        print(list(model_dict.keys()))
        current_model_dict = model_dict[list(model_dict.keys())[0]]
        model = current_model_dict["model"]
        config = current_model_dict["config"]
        config[DATALOADER][CONFIG_DEAD_LEAVES] = dict(
            blur_kernel_half_size=[0, 0],
            ds_factor=1,
            noise_stddev=[0., 50.]
        )
        config[DATALOADER][SIZE] = (512, 512)
        config[DATALOADER][BATCH_SIZE][VALIDATION] = 4
        dataloader = get_data_loader(config, frozen_seed=42)
        print(config)
        output_dir = Path(args.output_dir)/(config[NAME] + "_" + config[PRETTY_NAME])
        output_dir.mkdir(parents=True, exist_ok=True)
        infer(model, dataloader[VALIDATION], config, device, output_dir)


if __name__ == "__main__":
    infer_main(sys.argv[1:])

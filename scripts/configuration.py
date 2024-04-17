from pathlib import Path

NB_ID = "blind-deblurring-from-synthetic-data"  # This will be the name which appears on Kaggle.
GIT_USER = "balthazarneveu"  # Your git user name
GIT_REPO = "blind-deblurring-from-synthetic-data"  # Your current git repo
# Keep free unless you need to acess kaggle datasets. You'll need to modify the remote_training_template.ipynb.
KAGGLE_DATASET_LIST = [
    "balthazarneveu/deadleaves-div2k-512",  # Deadleaves classic
    "balthazarneveu/deadleaves-primitives-div2k-512",  # Deadleaves with extra primitives
    "balthazarneveu/motion-blur-kernels",  # Motion blur kernels
    "joe1995/div2k-dataset",
]
WANDBSPACE = "perceptual-denoiser"
TRAIN_SCRIPT = "scripts/train.py"  # Location of the training script

ROOT_DIR = Path(__file__).parent
OUTPUT_FOLDER_NAME = "__output"
INFERENCE_FOLDER_NAME = "__inference"

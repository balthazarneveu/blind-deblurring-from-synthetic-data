from pathlib import Path

NB_ID = "blind-deblurring-from-synthetic-data"  # This will be the name which appears on Kaggle.
GIT_USER = "balthazarneveu"  # Your git user name
GIT_REPO = "blind-deblurring-from-synthetic-data"  # Your current git repo
# Keep free unless you need to acess kaggle datasets. You'll need to modify the remote_training_template.ipynb.
KAGGLE_DATASET_LIST = []
WANDBSPACE = "deblur-from-deadleaves"
TRAIN_SCRIPT = "scripts/train.py"  # Location of the training script

ROOT_DIR = Path(__file__).parent
OUTPUT_FOLDER_NAME = "__output"
INFERENCE_FOLDER_NAME = "__inference"

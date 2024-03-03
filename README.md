# Blind deblurring from synthetic data
MVA project 2024 on [image restoration](https://delires.wp.imt.fr/)

- Jamy Lafenetre
- Balthazar Neveu

------

[Synthetic images as a regularity prior for image
restoration neural networks](https://hal.science/hal-03186499/file/papier_SSVM%20%281%29.pdf) by 
Raphaël Achddou, Yann Gousseau, Saïd Ladjal


------

## Setup

```bash
git clone https://github.com/balthazarneveu/blind-deblurring-from-synthetic-data.git
cd blind-deblurring-from-synthetic-data
pip install -e .
pip install interactive-pipe
pip install batch-processing
```

-------

## Training

### Local training
```bash
python scripts/train.py -e 1000
```
### Remote training
:key: After setting up your kaggle credentials (`scripts/__kaggle_login.py` as explained [here](https://github.com/balthazarneveu/mva_pepites?tab=readme-ov-file#remote-training))

```bash
python scripts/remote_training.py -e 1000 -u username -p
```
### Monitoring and tracking
Available on [Weights and Biases](https://wandb.ai/balthazarneveu/deblur-from-deadleaves)

## Live inference
Compare several models with a live inference
```bash
python scripts/interactive_inference_synthetic.py -e 1000 1001
```
import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
import torch
# from pytorch_lightning.plugins import DDPPlugin


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

# data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

data_path = '/content/mla_project/images'  #change this to your dataset path in colab if needed

has_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
data = VAEDataset(
    data_path=data_path,
    train_batch_size=16,
    val_batch_size=16,
    patch_size=(128, 128),
    num_workers=4,
    # pin_memory=len(config['trainer_params']['gpus']) != 0
    pin_memory=has_gpu
)


data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                #  strategy=DDPPlugin(find_unused_parameters=False),
                # strategy={"type": "ddp", "find_unused_parameters": False},
                strategy="ddp",                # just the name
                # accelerator="gpu",
                # devices=torch.cuda.device_count(),
                 **config['trainer_params'])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)
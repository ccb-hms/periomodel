""" Training script for perio1 model """
# %% Imports
import os
import pandas as pd
import glob
import logging
from pathlib import Path

# PyTorch modules
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateFinder, LearningRateMonitor

# PerioModel import
import periomodel
from periomodel.imageproc import ImageData, validate_image_data
from periomodel.torchdataset import DatasetFromDF, TorchDataset
from periomodel.models.perio1 import PerioModel

from IPython.display import display, HTML


# %% GPU configurations
is_cuda = torch.cuda.is_available()
print(f'CUDA available: {is_cuda}')
print(f'Number of GPUs found:  {torch.cuda.device_count()}')

if is_cuda:
    print(f'Current device ID:     {torch.cuda.current_device()}')
    print(f'GPU device name:       {torch.cuda.get_device_name(0)}')
    print(f'CUDNN version:         {torch.backends.cudnn.version()}')
    device_str = 'cuda:0'
    torch.cuda.empty_cache()
else:
    device_str = 'cpu'
device = torch.device(device_str)
print()
print(f'Device for model training/inference: {device}')
# Set precision
torch.set_float32_matmul_precision('medium')

# %% Model parameters

params = {'max_epochs': 6000,
          'batch_size': 128,
          'augment': 'augment_2',
          'dataset': 'periodata_240415',
          'learning_rate': 0.0002,
          'max_im_size': 2500,
          'num_workers': 8,
          'im_size': 512,
          'im_mean': 0,
          'im_std': 1,
          'hist_eq': True,
          'gray': True,
          'n_hidden': 128,
          'model': 'xrv',
          'model_name': 'periocl3_128',
          'model_version': 4,
          'checkpoint_every_n_epoch': 100,
          'checkpoint_monitor': 'val_loss',
          'check_val_every_n_epoch': 2,
          'save_top_k_epochs': 20,
          'file_col': 'file',
          'label_col': 'cl3',
          'comment': 'validation',
          'date': 240430}

# %% Directories and parameters
# data_dir = os.path.normpath('/home/andreas/data/dcmdata')
data_dir = os.environ.get('DATA_ROOT')

image_dir = os.path.join(data_dir, 'dataset', params.get('dataset'))
model_dir = os.path.join(data_dir, 'model')
checkpoint_dir = os.path.join(model_dir,
                              params.get('model_name'),
                              f'version_{params.get("model_version")}',
                              'checkpoints')
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

# Load image data frame
patient_split_data_file = f'{params.get("dataset")}_split_ide.parquet'
data_file = os.path.join(image_dir, patient_split_data_file)
df = pd.read_parquet(data_file)

# Replace the image directory
df = df.assign(fs_file=df['file'].apply(lambda x: os.path.join(image_dir, os.path.basename(x))))

# Get the labels for the model and the disease classes
label_list = sorted(list(df.get(params.get('label_col')).unique()))
disease_list = [list(df.loc[df[params.get('label_col')] == l, 'disease'].unique()) for l in label_list]

# %% Logging

log_dir = os.path.join(model_dir, params.get('model_name'), f'version_{params.get("model_version")}', 'log')
Path(log_dir).mkdir(parents=True, exist_ok=True)
log_file = os.path.join(log_dir, 'train.log')

dtfmt = '%y%m%d-%H:%M'
logfmt = '%(asctime)s-%(name)s-%(levelname)s-%(message)s'

logging.basicConfig(filename=log_file,
                    filemode='w',
                    level=logging.INFO,
                    format=logfmt,
                    datefmt=dtfmt)

logger = logging.getLogger(name=__name__)

# Add additional information to the params dictionary and log it
params.update({'model_dir': model_dir,
               'checkpoint_dir': checkpoint_dir,
               'label_list': label_list,
               'disease_list': disease_list,
               'data_file': data_file})

logger.info(f'Parameters: {params}')

# TensorBoard Logger
tb_logger = TensorBoardLogger(save_dir=model_dir,
                              name=params.get('model_name'),
                              version=params.get('model_version'))

# %% Callbacks

chkpt = ModelCheckpoint(dirpath=checkpoint_dir,
                        filename='perio-{epoch}',
                        monitor=params.get('checkpoint_monitor'),
                        mode='min',
                        save_last=True,
                        every_n_epochs=params.get('checkpoint_every_n_epoch'),
                        save_on_train_epoch_end=True,
                        save_top_k=params.get('save_top_k_epochs'))

lr_monitor = LearningRateMonitor(logging_interval='epoch',
                                 log_momentum=True)

# %% Create the data sets
tds = TorchDataset(data=df,
                   file_col='fs_file',
                   label_col=params.get('label_col'),
                   max_im_size=params.get('max_im_size'),
                   im_size=params.get('im_size'),
                   im_mean=params.get('im_mean'),
                   im_std=params.get('im_std'),
                   hist_eq=params.get('hist_eq'),
                   gray=params.get('gray'))


# The data sets
train_idx_set = set(df.loc[df.get('dset') == 'train'].index)
train_set = tds.create_dataset_from_index(index_set=train_idx_set,
                                          augment=params.get('augment'))
print(f'Samples in training set: {len(train_set)}')

val_idx_set = set(df.loc[df.get('dset') == 'val'].index)
val_set = tds.create_dataset_from_index(index_set=val_idx_set,
                                        augment=None)
print(f'Samples in validation set: {len(val_set)}')

test_idx_set = set(df.loc[df.get('dset') == 'test'].index)
test_set = tds.create_dataset_from_index(index_set=test_idx_set,
                                         augment=None)
print(f'Samples in test set: {len(test_set)}')

print()
display(params)

# %% Run the model training. Everything down from here can be a loop.

# seed_everything(42, workers=True)
perio_model = PerioModel(train_set=train_set,
                         val_set=val_set,
                         test_set=test_set,
                         batch_size=params.get('batch_size'),
                         num_workers=params.get('num_workers'),
                         model_name=params.get('model'),
                         n_outputs=len(params.get('label_list')),
                         n_hidden=params.get('n_hidden'),
                         lr=params.get('learning_rate'))

perio_trainer = Trainer(max_epochs=params.get('max_epochs'),
                        default_root_dir=model_dir,
                        callbacks=[chkpt, lr_monitor],
                        logger=tb_logger,
                        check_val_every_n_epoch=params.get('check_val_every_n_epoch'),
                        deterministic=False)

# %% Run the training
perio_trainer.fit(perio_model)
# When all done, run the test from the latest checkpoint
chk_file_list = sorted(glob.glob(os.path.join(checkpoint_dir, 'perio-epoch*.ckpt')))
if len(chk_file_list) > 0:
    chk_file = chk_file_list[-1]
    perio_trainer.test(perio_model, ckpt_path=chk_file, verbose=True)
else:
    logger.warning(f'No checkpoint found in {checkpoint_dir}.')

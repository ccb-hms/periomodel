"""
Periodontal Disease Model Ver 1
Andreas Werdich
Center for Computational Biomedicine
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
# PyTorch libraries
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import torchmetrics.classification as tmc
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

# Lightning framework
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, STEP_OUTPUT, OptimizerLRScheduler
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch import LightningModule

# https://mlmed.org/torchxrayvision/
import torchxrayvision as xrv

logger = logging.getLogger(__name__)


def performance_metrics(metric_dict, logits, target, metric_prefix='train'):
    """
    Calculate performance metrics for given predictions and targets.
    Parameters:
    -   metric_dict (dict): A dictionary of metric names and corresponding metric functions.
    -   logits (torch.Tensor): The predicted logits.
    -   target (torch.Tensor): The target values.
    -   metric_prefix (str, optional): The prefix to be added to each metric name in the output dictionary. Defaults to 'train'.
    Returns:
    -   dict: A dictionary containing the performance metrics with their corresponding names.
    """
    preds = nn.Softmax(dim=1)(logits)
    performance_dict = {}
    for metric_name, metric in metric_dict.items():
        performance_dict.update({f'{metric_prefix}_{metric_name}': metric(preds=preds, target=target)})
    return performance_dict


def average_performance_metrics(step_metrics_list, decimals=3):
    """
    Calculate the average performance metrics for a given list of step metrics.
    Parameters:
    - step_metrics_list (list): A list of dictionaries, where each dictionary represents the performance metrics for a particular step.
    - decimals (int, optional): Number of decimal places to round the average values to. Default is 3.
    Returns:
    - average_metrics (dict): A dictionary containing the average values of each performance metric.
    """
    average_metrics = {}
    if len(step_metrics_list) > 0:
        for metric in step_metrics_list[0].keys():
            metric_value = torch.stack([x.get(metric) for x in step_metrics_list])
            # Remove any zero values before averaging
            metric_value = metric_value[metric_value.nonzero().squeeze()]
            metric_value = metric_value.mean().detach().cpu().numpy().round(decimals)
            average_metrics.update({metric: metric_value})
    return average_metrics


class FineTuneLearningRateFinder(LearningRateFinder):
    """
    FineTuneLearningRateFinder is a class that extends the LearningRateFinder class.
    It is used to find the optimal learning rate for fine-tuning a model during training.
    Attributes:
        milestones (List[int]): A list of epoch numbers at which the learning rate should be evaluated.
    """

    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


class ResNetModels:
    """
    This class provides methods to create different disease models based on the ResNet architecture.
    Parameters:
    -   n_outputs (int): Number of output classes.
    -   n_hidden (int): Number of hidden units in the fully connected layers.
    Methods:
    -   xrv_model(): Returns a disease model with input size 512 x 512.
    -   resnet50_model(): Returns a disease model with input size 224 x 224.
    """

    def __init__(self, n_outputs=3, n_hidden=512):
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden

    def xrv_model(self):
        """ Input size is 512 x 512 """
        model = xrv.models.ResNet(weights="resnet50-res512-all")
        model = model.model
        model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=self.n_hidden),
            nn.ReLU(),
            nn.Linear(in_features=self.n_hidden, out_features=self.n_outputs))
        return model

    def resnet50_model(self):
        """ Input size is 224 x 224 """
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.n_outputs))
        return model


class PerioModel(LightningModule):
    def __init__(self,
                 train_set,
                 val_set,
                 batch_size,
                 num_workers,
                 test_set=None,
                 model_name: str = 'xrv',
                 n_outputs: int = 3,
                 n_hidden: int = 512,
                 lr: float = 0.001):
        super().__init__()
        self.save_hyperparameters()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.lr = lr
        self.decimals = 5

        # Performance metrics
        self.metrics = nn.ModuleDict({
            'accuracy': tmc.MulticlassAccuracy(num_classes=n_outputs, average='micro'),
            'precision': tmc.MulticlassPrecision(num_classes=n_outputs, average='macro'),
            'recall': tmc.MulticlassRecall(num_classes=n_outputs, average='macro'),
            'f1': tmc.MulticlassF1Score(num_classes=n_outputs, average='macro'),
            'auroc': tmc.MulticlassAUROC(num_classes=n_outputs, average='macro')
        })
        self.train_step_metrics_list = []
        self.val_step_metrics_list = []
        self.test_step_metrics_list = []

        # Initialize the model
        self.model = self._initialize_model()

        # Define the loss function
        self.criterion = nn.CrossEntropyLoss()

    def _initialize_model(self) -> nn.Module:
        if self.model_name == 'resnet50':
            resnet_models = ResNetModels(n_outputs=self.n_outputs, n_hidden=self.n_hidden)
            model = resnet_models.resnet50_model()
        elif self.model_name == 'xrv':
            resnet_models = ResNetModels(n_outputs=self.n_outputs, n_hidden=self.n_hidden)
            model = resnet_models.xrv_model()
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")
        return model

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dl = DataLoader(dataset=self.train_set,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=True,
                        pin_memory=True)
        return dl

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(dataset=self.val_set,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=False,
                        pin_memory=True)
        return dl

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dl = DataLoader(dataset=self.test_set,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=False,
                        pin_memory=True)
        return dl

    def loss_fn(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        logits = self(x)
        train_loss = self.loss_fn(logits, y)
        performance_dict = performance_metrics(metric_dict=self.metrics,
                                               logits=logits,
                                               target=y,
                                               metric_prefix='train')
        train_step_metrics = {'train_loss': train_loss}
        train_step_metrics.update(performance_dict)
        self.train_step_metrics_list.append(train_step_metrics)
        return train_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        logits = self(x)
        val_loss = self.loss_fn(logits, y)
        performance_dict = performance_metrics(metric_dict=self.metrics,
                                               logits=logits,
                                               target=y,
                                               metric_prefix='val')
        val_step_metrics = {'val_loss': val_loss}
        val_step_metrics.update(performance_dict)
        self.val_step_metrics_list.append(val_step_metrics)
        return val_loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        logits = self(x)
        test_loss = self.loss_fn(logits, y)
        performance_dict = performance_metrics(metric_dict=self.metrics,
                                               logits=logits,
                                               target=y,
                                               metric_prefix='test')
        test_step_metrics = {'test_loss': test_loss}
        test_step_metrics.update(performance_dict)
        self.test_step_metrics_list.append(test_step_metrics)
        return test_loss

    def predict_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Any:
        image, label = batch
        output = self.forward(image)
        return output

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.AdamW(params=self.parameters(), lr=self.lr)
        return opt

    def on_train_epoch_end(self) -> None:
        if len(self.train_step_metrics_list) > 0:
            epoch_train_metrics = average_performance_metrics(step_metrics_list=self.train_step_metrics_list,
                                                              decimals=self.decimals)
            self.log_dict(epoch_train_metrics, prog_bar=False)
        self.train_step_metrics_list.clear()

    def on_validation_epoch_end(self) -> None:
        if len(self.val_step_metrics_list) > 0:
            epoch_val_metrics = average_performance_metrics(step_metrics_list=self.val_step_metrics_list,
                                                            decimals=self.decimals)
            # Manually log learning rate
            epoch_val_metrics['val_lr'] = self.lr
            self.log_dict(epoch_val_metrics, prog_bar=True)
        self.val_step_metrics_list.clear()

    def on_test_epoch_end(self) -> None:
        if len(self.test_step_metrics_list) > 0:
            epoch_test_metrics = average_performance_metrics(step_metrics_list=self.test_step_metrics_list,
                                                             decimals=self.decimals)
            self.log_dict(epoch_test_metrics, prog_bar=True)
            logger.info(f'test: {epoch_test_metrics}')
        self.test_step_metrics_list.clear()




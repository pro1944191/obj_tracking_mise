"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os

import torch

from torchvision import transforms

import data_setup, model_engine, model_builder, utils

# Adding Hydra support
import logging

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from hydra.utils import to_absolute_path


# Logger
log = logging.getLogger(__name__)


def train(conf: DictConfig) -> None:
    # Setup hyperparameters
    #NUM_EPOCHS = conf.train.num_epochs
    #BATCH_SIZE = conf.train.batch_size
    #HIDDEN_UNITS = conf.train.hidden_hunits
    #LEARNING_RATE = conf.train.learning_rate

    # Setup directories
    #train_dir = to_absolute_path(conf.dataset.train_dir)
    #test_dir = to_absolute_path(conf.dataset.test_dir)
    video_dir = to_absolute_path(conf.data_to_process.video_dir)
    image_dir = to_absolute_path(conf.data_to_process.image_dir)

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: change transforms if needed
    # Create transforms
    #data_transform = transforms.Compose(
    #    [transforms.Resize((conf.dataset.resize_size, conf.dataset.resize_size)), 
    #     transforms.ToTensor()]
    #)

    # Create DataLoaders with help from data_setup.py
    #train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    #    train_dir=train_dir,
    #    test_dir=test_dir,
    #    transform=data_transform,
    #    batch_size=BATCH_SIZE,
    #)

    # TODO: Change it if you need to
    # Create model with help from model_builder.py
    #model = model_builder.YourModel(
    #    input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)
    #).to(device)

    # TODO: Change it if you need to
    # Set loss and optimizer
    #loss_fn = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start training with help from engine.py
    #model_engine.train(
    #    model=model,
    #    train_dataloader=train_dataloader,
    #    test_dataloader=test_dataloader,
    #    loss_fn=loss_fn,
    #    optimizer=optimizer,
    #    epochs=NUM_EPOCHS,
    #    device=device,
    #)

    # Save the model with help from utils.py
    #utils.save_model(
    #    model=model,
    #    target_dir=to_absolute_path("models"),
    #    model_name=conf.train.model_name,
    #)


@hydra.main(config_path="../conf", config_name="config")
def main(conf: DictConfig):
    log.info(OmegaConf.to_yaml(conf))
    train(conf)


if __name__ == "__main__":
    main()

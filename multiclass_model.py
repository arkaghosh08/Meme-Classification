import os
from pickle import load, dump
import pytorch_lightning as pl
from sympy import N
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class DenseNetTransferLearning(pl.LightningModule):
    def __init__(self, learning_rate=2e-4, num_target_classes=1145, model_cache_dir="./model_cache"):
        super().__init__()
        print("Initializing DenseNet model for transfer learning...")
        self.__learning_rate = learning_rate
        self.__num_target_classes = num_target_classes
        self.__model_cache_dir = model_cache_dir

        # Ensure the model cache directory exists
        if not os.path.exists(self.__model_cache_dir):
            os.makedirs(self.__model_cache_dir)
            print(f"Created model cache directory at {self.__model_cache_dir}")

        # Set the environment variable for the Torch Hub directory to store pre-trained models locally
        os.environ["TORCH_HOME"] = self.__model_cache_dir

        # Load the pretrained DenseNet model
        print("Loading pretrained DenseNet model from local cache (or downloading if not available)...")
        self.__feature_extractor = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.__feature_extractor = nn.Sequential(*list(self.__feature_extractor.children())[:-1])  # Remove the classification layer
        print("Feature extractor loaded.")

        # Freeze the layers of the pretrained model
        for param in self.__feature_extractor.parameters():
            param.requires_grad = False
        print("Feature extractor frozen.")

        # Add a new classifier layer for the number of target classes
        print(f"Adding classifier for {self.__num_target_classes} classes.")
        self.__classifier = nn.Linear(in_features=1024, out_features=self.__num_target_classes)  # Adjust in_features for DenseNet
        self.__criterion = nn.CrossEntropyLoss()
        self.__global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.__vloss = []
        self.__tloss = []

    def forward(self, x):
        x = self.__feature_extractor(x)
        x = self.__global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.__classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.__criterion(outputs, labels)
        self.__tloss.append(loss.item())
        self.log('train_loss', loss, prog_bar=True)
        print(f"\nTraining step {batch_idx}: Loss = {loss.item()}")
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.__criterion(outputs, labels)
        self.__vloss.append(loss.item())
        self.log('val_loss', loss, prog_bar=True)
        print(f"\nValidation step {batch_idx}: Loss = {loss.item()}")
        return loss

    def configure_optimizers(self):
        print(f"\nConfiguring Adam optimiser with learning rate {self.__learning_rate}.")
        self.__optimiser = optim.Adam(self.parameters(), lr=self.__learning_rate)
        self.__scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.__optimiser, 'min', patience=3, factor=0.5)
        return {'optimizer': self.__optimiser, 'lr_scheduler': self.__scheduler, 'monitor': 'val_loss'}
    
    def optimiser_state_dict(self):
        return self.__optimiser.state_dict()
    
    def log_losses(self):
        with open('model_losses.bin', 'wb') as f:
            dump({'validation': self.__vloss, 'training': self.__tloss}, f)

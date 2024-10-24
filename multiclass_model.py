import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import models
import os

class DenseNetTransferLearning(pl.LightningModule):
    def __init__(self, learning_rate=2e-4, num_target_classes=1145, model_cache_dir="./model_cache"):
        super().__init__()
        print("Initializing DenseNet model for transfer learning...")
        self.learning_rate = learning_rate
        self.num_target_classes = num_target_classes
        self.model_cache_dir = model_cache_dir

        # Ensure the model cache directory exists
        if not os.path.exists(self.model_cache_dir):
            os.makedirs(self.model_cache_dir)
            print(f"Created model cache directory at {self.model_cache_dir}")

        # Set the environment variable for the Torch Hub directory to store pre-trained models locally
        os.environ["TORCH_HOME"] = self.model_cache_dir

        # Load the pretrained DenseNet model
        print("Loading pretrained DenseNet model from local cache (or downloading if not available)...")
        self.feature_extractor = models.densenet121(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])  # Remove the classification layer
        print("Feature extractor loaded.")

        # Freeze the layers of the pretrained model
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        print("Feature extractor frozen.")

        # Add a new classifier layer for the number of target classes
        print(f"Adding classifier for {self.num_target_classes} classes.")
        self.classifier = nn.Linear(in_features=1024, out_features=self.num_target_classes)  # Adjust in_features for DenseNet
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        print(f"Training step {batch_idx}: Loss = {loss.item()}")
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)
        print(f"Validation step {batch_idx}: Loss = {loss.item()}")
        return loss

    def configure_optimizers(self):
        print(f"Configuring Adam optimizer with learning rate {self.learning_rate}.")
        return optim.Adam(self.parameters(), lr=self.learning_rate)

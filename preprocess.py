import numpy as np
import pandas as pd
from PIL import Image
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision import transforms

class DatasetLoader(Dataset):
    def __init__(self, df_path, transform=transforms.ToTensor()):
        print("Loading dataset...")
        self.__df = pd.read_parquet(df_path)
        print(f"Dataset loaded with {len(self.__df)} entries.")

        self.__df['path'] = self.__df['path'].str.replace('/dataset/', './datasets/')
        self.__transform = transform
        self.__numClasses = self.__get_numClasses()
        self.__X, self.__y = self.__get_X_y()

    def __get_numClasses(self):
        num_classes = self.__df["template_name"].nunique()
        print(f'Number of unique classes: {num_classes}')
        return num_classes

    def __get_X_y(self):
        print("Encoding template names to template IDs...")
        self.__encodeTemplates()
        X = self.__df['path']
        y = self.__df['template_id'].astype(float)
        print("Encoding completed.")
        return X, y

    def __encodeTemplates(self):
        self.__df['template_id'] = LabelEncoder().fit_transform(self.__df['template_name'])

    def __len__(self):
        return len(self.__X)

    def __getitem__(self, idx):
        try:
            image_path = self.__X.iloc[idx]
            img = Image.open(image_path)
            label = self.__y.iloc[idx]

            if self.__transform:
                img = self.__transform(img)
        except:
            print(f"Couldn't load image at index {idx}!")
            return None, None

        return img, label


class MemesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_df_path, test_size=0.2):
        super().__init__()
        self.data_df_path = data_df_path
        self.batch_size = batch_size
        self.test_size = test_size
        print(f"Initializing MemesDataModule with batch size {self.batch_size} and test size {self.test_size}.")

        # Augmentation policy for training set
        self.augmentation = transforms.Compose([
              transforms.Resize(size=(224,224)),
              transforms.RandomRotation(degrees=(0, 10)),
              transforms.ColorJitter(brightness=0.1, contrast=0.1),
              transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
              transforms.ToTensor(),
              transforms.Normalize([0.5325, 0.4980, 0.4715],[0.3409, 0.3384, 0.3465])
        ])
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose([
              transforms.Resize(size=(224,224)),
              transforms.ToTensor(),
              transforms.Normalize([0.5325, 0.4980, 0.4715],[0.3409, 0.3384, 0.3465])
        ])

    def prepare_data(self):
        print("Preparing data...")

    def setup(self, stage=None):
        print(f"Setting up data for stage: {stage}")
        # build dataset
        dataset = DatasetLoader(df_path=self.data_df_path)

        # Stratified Sampling for train and val
        train_idx, validation_idx = train_test_split(np.arange(len(dataset)),
                                                    test_size=self.test_size,
                                                    shuffle=True,
                                                    stratify=dataset.__y)

        print(f"Train/validation split: {len(train_idx)} train samples, {len(validation_idx)} validation samples.")

        # Subset dataset for train and val
        self.train = Subset(dataset, train_idx)
        self.val = Subset(dataset, validation_idx)

        self.train.dataset.transform = self.augmentation
        self.val.dataset.transform = self.transform

    def train_dataloader(self):
        print("Creating training dataloader...")
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=my_collate_fn)

    def val_dataloader(self):
        print("Creating validation dataloader...")
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4, collate_fn=my_collate_fn)

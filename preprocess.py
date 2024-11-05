from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import numpy as np
import pandas as pd
from PIL import Image
from pickle import load, dump
import psutil
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision import transforms


class DatasetLoader(Dataset):
    def __init__(self, df_path, dataset_limit, transform=None):
        print("Loading dataset...")
        self.__df = pd.read_parquet(df_path)
        if not isinstance(dataset_limit, int):
            if not (isinstance(dataset_limit, float) and 0 <= dataset_limit <= 1):
                dataset_limit = 1
            dataset_limit *= len(self.__df)
            dataset_limit = int(dataset_limit)
        self.__df = self.__df[:dataset_limit]
        print(f"Dataset loaded with {len(self.__df)} entries.")
        if transform is None:
            self.__transform = transforms.Compose([
                transforms.ToTensor()
            ])
        self.__numClasses = self.__get_numClasses()
        self.__X, self.__y = self.get_X_y()
        self.__cached = False

    def __encodeTemplates(self):
        self.__df['template_id'] = LabelEncoder().fit_transform(self.__df['template_name'])

    def __len__(self):
        return len(self.__X)

    def __getitem__(self, idx):
        if self.__cached:
            with open('images_cache.bin', 'rb') as f:
                r = load(f)
            img, label = r[idx]
        else:
            try:
                image_path = self.__X.iloc[idx]
                img = Image.open(image_path)
                label = torch.tensor(self.__y.iloc[idx], dtype=torch.long)

                if self.__transform:
                    img = self.__transform(img)
            except Exception as e:
                print(f"Couldn't load image at index {idx}: {e}!")
                return None, None

        return img, label

    def setTransformation(self, transform):
        self.__transform = transform

    def __get_numClasses(self):
        num_classes = self.__df["template_name"].nunique()
        print(f'Number of unique classes: {num_classes}')
        return num_classes

    def get_X_y(self):
        print("Encoding template names to template IDs...")
        self.__encodeTemplates()
        X = self.__df['path']
        y = self.__df['template_id'].astype(int)
        print("Encoding completed.")
        return X, y
    
    def load_image_and_label(self, i):
        with Image.open(self.__X.iloc[i]) as img:
            img_t = self.__transform(img)
        del img
        label = torch.tensor(self.__y.iloc[i], dtype=torch.long)
        print(f'Loaded image,label {i}')

        mem = psutil.virtual_memory()
        if mem.available < 0.05 * mem.total:
            gc.collect()
            torch.cuda.empty_cache()
        return img_t, label
    
    def cache_items(self):
        arr = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.load_image_and_label, i): i for i in range(len(self.__df))}
            for future in as_completed(futures):
                i = futures[future]
                arr[i] = future.result()
        with open('images_cache.bin', 'wb') as f:
            dump(arr, f)
        print('Dumped images cache successfully!')
        self.__cached = True


class MemesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, df_path, test_size=0.2, lazy_load=True, cache=False, dataset_limit=1.0):
        super().__init__()
        self.__df_path = df_path
        self.__batch_size = batch_size
        self.__test_size = test_size
        print(f"Initializing MemesDataModule with batch size {self.__batch_size} and test size {self.__test_size}.")

        self.__dflimit = dataset_limit

        self.__dataset = None
        if not lazy_load:
            self.__dataset = DatasetLoader(df_path=self.__df_path, dataset_limit=self.__dflimit)
            if cache:
                self.__dataset.cache_items()

        # Augmentation policy for training set
        self.__augmentation = transforms.Compose([
              transforms.Resize(size=(224,224)),
              transforms.RandomRotation(degrees=(0, 10)),
              transforms.ColorJitter(brightness=0.1, contrast=0.1),
              transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
              transforms.ToTensor(),
              transforms.Normalize([0.5325, 0.4980, 0.4715],[0.3409, 0.3384, 0.3465])
        ])
        # Preprocessing steps applied to validation and test set.
        self.__transform = transforms.Compose([
              transforms.Resize(size=(224,224)),
              transforms.ToTensor(),
              transforms.Normalize([0.5325, 0.4980, 0.4715],[0.3409, 0.3384, 0.3465])
        ])
    
    def prepare_data(self):
        print("Preparing data...")

    def setup(self, stage=None):
        print(f"Setting up data for stage: {stage}")
        # build dataset
        if self.__dataset is None:
            self.__dataset = DatasetLoader(df_path=self.__df_path, dataset_limit=self.__dflimit)

        # Stratified Sampling for train and val
        train_idx, validation_idx = train_test_split(np.arange(len(self.__dataset)),
                                                    test_size=self.__test_size,
                                                    shuffle=True,
                                                    random_state=42,
                                                    stratify=self.__dataset.get_X_y()[1])

        print(f"Train/validation split: {len(train_idx)} train samples, {len(validation_idx)} validation samples.")

        # Subset dataset for train and val
        self.__train = Subset(self.__dataset, train_idx)
        self.__val = Subset(self.__dataset, validation_idx)

        self.__train.dataset.setTransformation(self.__augmentation) # type: ignore
        self.__val.dataset.setTransformation(self.__transform) # type: ignore

    def train_dataloader(self):
        print("\nCreating training dataloader...")
        d = DataLoader(self.__train, batch_size=self.__batch_size, shuffle=True, num_workers=4, collate_fn=my_collate_fn, persistent_workers=True, pin_memory=True)
        print()
        return d

    def val_dataloader(self):
        print("\nCreating validation dataloader...")
        d = DataLoader(self.__val, batch_size=self.__batch_size, num_workers=4, collate_fn=my_collate_fn, persistent_workers=True, pin_memory=True)
        print()
        return d
    
    def dataset_size(self):
        try:
            return len(self.__dataset)
        except:
            return -1
        
    def save(self):
        if self.__dataset is None: return False
        print('Saving image cache...')
        self.__dataset.cache_items()
        print('Save completed!')

    
def my_collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return torch.Tensor, torch.Tensor
    return torch.utils.data.dataloader.default_collate(batch) # type: ignore

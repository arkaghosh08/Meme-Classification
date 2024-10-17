from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import pandas as pd
from PIL import Image

class MultiClass:
    def __init__(self, df_path, transform=transforms.ToTensor()):
        self.__df = pd.read_parquet(df_path)
        self.__df['path'] = self.__df['path'].str.replace('D:/Memes2024', '../../dataset/ImgFlipMemes')
        self.__transform = transform
        self.__numClasses = self.__get_numClasses()
        self.__X, self.__y = self.__get_X_y()

    def __get_numClasses(self):
        print(f'Number of unique classes: {self.__df["template_name"].nunique()}')

    def __get_X_y(self):
        self.__encodeTemplates()
        X = self.__df['path']
        y = self.__df['template_id'].astype(float)
        return X, y

    def __encodeTemplates(self):
        self.__df['template_id'] = LabelEncoder().fit_transform(self.__df['template_names'])

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
            print('Couldn\'t load image!')
            return None, None
        
        return img, label
    

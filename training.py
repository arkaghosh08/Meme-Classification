from multiclass_model import *
from preprocess import *

data_module = MemesDataModule(batch_size=32, data_df_path="../../datasets/filtered_dataset.parquet")
model = DenseNetTransferLearning(num_target_classes=1145)

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, data_module)

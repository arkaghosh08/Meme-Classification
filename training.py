from multiclass_model import *
from preprocess import *
import torch

if __name__ == '__main__':
    data_module = MemesDataModule(batch_size=64, df_path="../../datasets/filtered_dataset.parquet", dataset_limit=.0075)
    trainer = pl.Trainer(max_epochs=3, default_root_dir='./training_logs', precision='bf16-mixed')

    model = DenseNetTransferLearning()
    trainer.fit(model, data_module)

    model.log_losses()
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': model.optimiser_state_dict()
    }, 'model_save.pth')
    print('Model saved successfully!')

import matplotlib.pyplot as plt
from multiclass_model import DenseNetTransferLearning
from pickle import load
from preprocess import MemesDataModule
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import torch

if __name__ == '__main__':
    data_module = MemesDataModule(batch_size=64, df_path="../../datasets/filtered_dataset.parquet", dataset_limit=.0075)
    data_module.setup()

    # Load the model
    checkpoint = torch.load("model_save.pth")
    model = DenseNetTransferLearning()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode

    # Run on the validation set
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_module.val_dataloader():
            images, labels = batch
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    with open('model_losses.bin', 'rb') as f: r = load(f)
    
    train_losses = r['training']
    val_losses = r['validation']

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.savefig('losses.png')
    plt.show()

    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png')
    plt.show()

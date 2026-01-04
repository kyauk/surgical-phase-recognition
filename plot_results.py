
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def plot_confusion_matrix(y_true, y_pred, class_names, output_path="checkpoints/confusion_matrix.png"):
    """
    Plots a confusion matrix using seaborn and saves it as an image.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Normalize by row (ground truth)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")
    plt.close()

def plot_training_history(train_losses, val_losses, train_accs, val_accs, output_dir="checkpoints"):
    """
    Plots training and validation loss/accuracy curves and saves them.
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Header for Loss Plot
    plt.figure(figsize=(12, 5))
    
    # Loss Subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy Subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Acc')
    plt.plot(epochs, val_accs, 'r-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "training_curves.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(save_path)
    print(f"Training curves saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    print("This script is intended to be imported.")

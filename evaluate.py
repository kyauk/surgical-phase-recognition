import sys
import os
import torch
from torch.utils.data import DataLoader
from data_utils.dataloader import get_dataloaders
from models.model import SurgicalModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import BASE_DIR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from plot_results import plot_confusion_matrix

BATCH_SIZE=32

def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = SurgicalModel()
    model.to(device)
    
    # Load Weights
    weights_path = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
    if os.path.exists(weights_path):
         model.load_state_dict(torch.load(weights_path, map_location=device))
         print(f"Loaded weights from {weights_path}")
    else:
         print(f"WARNING: No weights found at {weights_path}")

    _, _, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    """ 
    ------------------------------
    STAGE 1: Getting Model's preds
    ------------------------------
    """
    print("Running Inference on Test Set...")
    all_preds = []
    all_labels = []

    # predict() function from train.py returns a tensor of predictions
    # But we need to iterate to get labels since predict() doesn't return them.
    # Let's do the manual loop here to get both efficiently, using the logic from train.predict
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, batch_preds = outputs.max(dim=2)
            all_preds.extend(batch_preds.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    """ 
    ------------------------------
    STAGE 2: Computing Evaluation Metrics
    ------------------------------
    """

    print("------------- TEST RESULTS ----------------")
    
    # Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {acc*100:.2f}%")

    # Classification Report (Precision, Recall, F1)
    # Get phase names from the dataset mapping
    phase_mapping = test_loader.dataset.phase2int
    # Sort names by index to match the integer output
    target_names = [name for name, i in sorted(phase_mapping.items(), key=lambda x: x[1])]
    
    print("Detailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # Confusion Matrix
    print("Saving Confusion Matrix to files...")
    cm_path = os.path.join(BASE_DIR, "checkpoints", "confusion_matrix.png")
    plot_confusion_matrix(all_labels, all_preds, target_names, output_path=cm_path)


if __name__ == "__main__":
    evaluate_model()



    



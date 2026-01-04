import sys
import os
# Add project root to path (for running script directly from train/ dir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.dataloader import get_dataloaders
from models.model import SurgicalModel
from tqdm import tqdm
from constants import BATCH_SIZE, ANNOTATIONS_DIR, SEQ_LEN
from plot_results import plot_training_history
import os
import torch
import torch.nn as nn
import torch.optim as optim

EPOCHS = 100
TRAIN_WEIGHTS_PATH = "best_model.pth"
SAVE_DIR = "checkpoints/"

def train():
    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"using device: {device}")
    
    # load data
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
   
    # load model
    model = SurgicalModel()
    print("Model Initialized")
    model.to(device)

    # define optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_val_acc = 0.0
    
    # History for plotting
    history_train_loss = []
    history_val_loss = []
    history_train_acc = []
    history_val_acc = []

    # Training Loop
    model.eval()
    with torch.no_grad():
        for inputs, labels in train_loader:
            print(f"Input shape: {inputs.shape}")
            outputs = model(inputs)
            print(f"Output shape: {outputs.shape}")
            print(f"Labels shape: {labels.shape}")
            break
    exit()
    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # transform output to be format it to how loss is expecting it, where num classes is 2nd dim (index 1)
            outputs = outputs.transpose(1,2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.numel()
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'loss': total_loss/(pbar.n+1), 'acc': 100.*correct/total})
            
        # End of Epoch Metrics
        epoch_train_loss = total_loss / len(train_loader)
        epoch_train_acc = 100. * correct / total
        history_train_loss.append(epoch_train_loss)
        history_train_acc.append(epoch_train_acc)

        scheduler.step()

        # Validation Step
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_accum = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # transpose for reasons as mentioned above
                loss_outputs = outputs.transpose(1, 2)
                loss = criterion(loss_outputs, labels)
                val_loss_accum += loss.item()
                
                _, predicted = loss_outputs.max(1)
                val_total += labels.numel()
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss_accum / len(val_loader)
        
        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc)
        
        print(f"Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}% @ EPOCH {epoch}")


        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(SAVE_DIR, TRAIN_WEIGHTS_PATH)
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model ({best_val_acc:.2f}%) to {save_path}")


    print("Training Complete")
    
    # Plot results
    plot_training_history(history_train_loss, history_val_loss, 
                          history_train_acc, history_val_acc, 
                          output_dir=SAVE_DIR)

    # Save Final Model
    final_path = os.path.join(SAVE_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved Final Model to {final_path}")
    

if __name__ == "__main__":
    train()



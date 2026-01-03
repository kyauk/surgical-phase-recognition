from data_utils.dataloader import get_dataloaders
from models.model import SequencingModel
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim


def train():
    EPOCHS = 100
    TRAIN_WEIGHTS_PATH = "best_model.pth"
    BATCH_SIZE = 32
    SAVE_DIR = "checkpoints/"
    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"using device: {device}")
    
    # load data
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
   
    # load model
    model = SequencingModel()
    print("Model Initialized")
    model.to(device)

    # define optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training Loop
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_val_acc = 0.0

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

        scheduler.step()

        # Validation Step
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).transpose(1,2)
                _, predicted = outputs.max(1)
                val_total += labels.numel()
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}% @ EPOCH {epoch}")

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(SAVE_DIR, TRAIN_WEIGHTS_PATH)
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model ({best_val_acc:.2f}%) to {save_path}")


    print("Training Complete")
    # Save Final Model
    final_path = os.path.join(SAVE_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved Final Model to {final_path}")
if __name__ == "__main__":
    train()



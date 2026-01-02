from data_utils.dataloader import get_dataloaders
from models.model import SequencingModel
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim


def train():
    EPS = 1e-8
    EPOCHS = 100
    TRAIN_FEATURES_PATH = "cached_features_train.pt"
    BATCH_SIZE = 32
    SAVE_DIR = "checkpoints/"
    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"using device: {device}")
    # load data
    train_loader, val_loader, test_loader = get_dataloaders()
   
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

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # transform output to be format it to how loss is expecting it, where num classes is 2nd dim
            outputs = outputs.transpose(1,2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    print("Training Complete")
if __name__ == "__main__":
    train()



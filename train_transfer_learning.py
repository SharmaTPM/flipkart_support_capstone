import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

# 1. Device Setup & Reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# 2. Preprocessing & Data Augmentation (ResNet-18 Requirements)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),  # Replicate 1 channel to 3
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load Canonical Fashion-MNIST Dataset
full_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# Carve stratified validation set (5,000 images) out of 60,000 train set
train_ds, val_ds = random_split(
    full_train, [55000, 5000], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# 4. Build Transfer Learning Model (Feature Extraction with Frozen Backbone)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze backbone parameters
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head for 10 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 5. Model Training Routine
def train_model(epochs=5):
    print("Training feature extraction head...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation Check
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = torch.argmax(model(imgs), dim=1)
                val_correct += (preds == lbls).sum().item()
        
        val_acc = val_correct / len(val_ds)
        print(f"Epoch {epoch}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {val_acc*100:.2f}%")

# 6. Evaluation Routine
def evaluate_test_set():
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(lbls.numpy())
    
    test_acc = (np.array(all_preds) == np.array(all_targets)).mean()
    print(f"\nHeld-Out Test Accuracy: {test_acc*100:.2f}%\n")
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES))
    
    cm = confusion_matrix(all_targets, all_preds)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    print("\n--- 10x10 CONFUSION MATRIX ---")
    print(cm_df)

# 7. Artifact Persistence & Export PNGs
def save_artifacts():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/sample_images", exist_ok=True)
    
    # Save Model Weights
    torch.save(model.state_dict(), "models/product_classifier.pt")
    print("Saved model weights to models/product_classifier.pt")
    
    # Export 5 Real PNG Sample Files
    raw_test = datasets.FashionMNIST(root="./data", train=False, download=True)
    sample_indices = [0, 1, 3, 6, 8]
    for idx in sample_indices:
        img, lbl = raw_test[idx]
        category_slug = CLASS_NAMES[lbl].lower().replace('/', '_').replace(' ', '_')
        out_path = f"data/sample_images/{idx:02d}_{category_slug}.png"
        img.save(out_path)
    print("Exported 5 sample PNG files to data/sample_images/")

if __name__ == "__main__":
    train_model(epochs=5)
    evaluate_test_set()
    save_artifacts()
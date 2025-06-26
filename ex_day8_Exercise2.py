model_names = ["google/vit-base-patch16-224",
               "microsoft/beit-base-patch16-224",
               "facebook/vit-mae-base"]
pretrained_model = model_names[0]

# (1) Data Loading and Preprocessing
from transformers import AutoImageProcessor
processor = AutoImageProcessor.from_pretrained(pretrained_model, use_fast=True)
image_size = processor.size["height"]

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

# Oxford-IIIT Pets Dataset
from torchvision.datasets import OxfordIIITPet
trainval_dataset = OxfordIIITPet(root='./data', split='trainval', transform=transform, download=True)
test_dataset = OxfordIIITPet(root='./data', split='test', transform=transform, download=True)
num_classes = len(trainval_dataset.classes)

# Split trainval into train and validation sets
train_size = int(0.8 * len(trainval_dataset))
val_size = len(trainval_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=64)

# Print the first sample
x, y = trainval_dataset[0]
print(f"Sample shape: {x.shape}, Label: {trainval_dataset.classes[y]}")

# (2) Training function
import torch.optim as optim

def train(train_loader, val_loader, model, optimizer, device):
    criterion = nn.CrossEntropyLoss()
    train_accs, val_accs = [], []

    best_acc = 0.0
    best_state = None

    for epoch in range(10):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            inputs = processor(images=list(x), return_tensors="pt", do_rescale=False)
            x, y = inputs['pixel_values'].to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        train_accs.append(correct / total)

        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                inputs = processor(images=list(x), return_tensors="pt", do_rescale=False)
                x, y = inputs['pixel_values'].to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
        val_accs.append(correct / total)

        print(f"Epoch {epoch+1}: Train Acc = {train_accs[-1]:.3f}, Val Acc = {val_accs[-1]:.3f}")
    return train_accs, val_accs, best_state

# (3) Definition of the model and optimizer
from torch import nn
from transformers import AutoModel
import torchinfo

class ViTClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes, freeze_encoder=True):
        super().__init__()
        self.vit = AutoModel.from_pretrained(pretrained_model)
        if freeze_encoder:
            for param in self.vit.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS]
        logits = self.classifier(cls_token)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTClassifier(pretrained_model, num_classes, freeze_encoder=True).to(device)
print(torchinfo.summary(model, input_size=(32, 3, 224, 224), row_settings=["var_names"],
                        col_names=["input_size", "output_size", "num_params"]))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# (4) Run training and plot the train/validation accuracy
import matplotlib.pyplot as plt
train_accs, test_accs, best_state = train(train_loader, val_loader, model, optimizer, device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Valid Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Classifier with {pretrained_model} (Params: {num_params})")
plt.legend()
plt.grid(True)
plt.show()

# (5) Evaluate on test set
model.load_state_dict(best_state)
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        inputs = processor(images=list(x), return_tensors="pt", do_rescale=False)
        x, y = inputs['pixel_values'].to(device), y.to(device)
        out = model(x)
        all_preds.append(out.argmax(1).cpu())
        all_labels.append(y.cpu())
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
print(f"Test Accuracy: {torch.sum(all_preds == all_labels).item() / len(all_labels):.3f}")

from sklearn.metrics import classification_report
print(classification_report(all_labels.numpy(), all_preds.numpy()))
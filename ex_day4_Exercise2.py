# (1) Data loading and splitting
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
data = fetch_covtype()
X = data.data
y = data.target - 1  # 0-indexing

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, 
                                                              stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, 
                                                  stratify=y_train_full, random_state=42)

# (2) Data preprocessing (standardization and conversion to tensors)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

import torch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# (3) Create DataLoader
from torch.utils.data import TensorDataset, DataLoader
batch_size = 256
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

# (4) Define MLP model
from torch import nn
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module('input', nn.Linear(input_dim, hidden_dims[0]))
        self.net.add_module('relu1', nn.ReLU())
        for i in range(1, len(hidden_dims)):
            self.net.add_module(f'hidden{i}', nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.net.add_module(f'relu{i+1}', nn.ReLU())
        self.net.add_module('output', nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        return self.net(x)

# (5) Training function
from sklearn.metrics import accuracy_score
def train_model(train_loader, val_loader, input_dim, hidden_dims, output_dim, 
                lr=0.001, epochs=100, device='cpu'):
    model = MLP(input_dim, hidden_dims, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
    
    return best_acc, best_state

# (6) Hyperparameter tuning
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hidden_dims = [(64,), (128,), (64, 32)]
learning_rates = [0.001, 0.01]
results = []

for h in hidden_dims:
    for lr in learning_rates:
        acc, state = train_model(train_loader, val_loader, X_train.shape[1], 
                                 h, 7, lr, epochs=100, device=device)
        results.append((h, lr, acc, state))
        print(f"Hidden={h}, LR={lr} => Val Acc={acc:.4f}")

# (7) Select the best model
best_h, best_lr, _, best_state = max(results, key=lambda x: x[2])
model = MLP(X_train.shape[1], best_h, 7).to(device)
model.load_state_dict(best_state)
model.eval()

# (8) Evaluate on test data
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
print(f"\nBest Hyperparameters: hidden={best_h}, lr={best_lr}")
print(f"Test Accuracy: {test_acc:.4f}")

from sklearn.metrics import classification_report
print(classification_report(all_labels, all_preds))
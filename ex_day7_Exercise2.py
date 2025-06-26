# (1) Data Loading
# Tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")

# Dataset
from torch.utils.data import RandomSampler

# IMDb Dataset
from datasets import load_dataset
dataset = load_dataset("imdb")
num_classes = len(dataset["train"].features["label"].names)
print(f"Number of classes: {num_classes}")
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=False)
tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
tokenized_trainval = tokenized["train"].train_test_split(test_size=0.2, seed=42)
train_set = tokenized_trainval["train"]
val_set = tokenized_trainval["test"]
test_set = tokenized["test"]
subset_size = len(train_set) // 2   # 1/2 of the training set is used
train_sampler = RandomSampler(train_set, num_samples=subset_size)

# MultiNLI Dataset
# from datasets import load_dataset
# dataset = load_dataset("multi_nli")
# num_classes = len(dataset["train"].features["label"].names)
# print(f"Number of classes: {num_classes}")
# def tokenize(batch):
#     return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, padding=False)
# tokenized = dataset.map(tokenize, batched=True)
# tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
# train_set = tokenized["train"]
# val_set = tokenized["validation_matched"]
# test_set = tokenized["validation_mismatched"]
# subset_size = len(train_set) // 30   # 1/30 of the training set is used
# train_sampler = RandomSampler(train_set, num_samples=subset_size)

# Print the first example
print("First example from the dataset:")
print("Sentence:", tokenizer.decode(train_set[0]["input_ids"]))
print("Label:", dataset["train"].features["label"].names[train_set[0]["label"]])

# (2) DataLoader with padding
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def collate_fn(batch):
    input_ids = pad_sequence([x["input_ids"] for x in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attn_mask = pad_sequence([x["attention_mask"] for x in batch], batch_first=True, padding_value=0)
    labels = torch.tensor([x["label"] for x in batch])
    return input_ids, attn_mask, labels

train_loader = DataLoader(train_set, batch_size=32, sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, collate_fn=collate_fn)

# (3) Model Definition
import torch.nn as nn

# MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)  # (B, L) -> (B, L, embed_dim)
        emb = emb.mean(dim=1)  # (B, L, embed_dim) -> (B, embed_dim)
        out = torch.relu(self.fc1(emb)) # (B, embed_dim) -> (B, hidden_dim)
        return self.fc2(out)  # (B, hidden_dim) -> (B, num_classes)

# RNN Classifier
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id)
        self.rnn = nn.RNN(embed_dim, hidden_dim, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids) # (B, L) -> (B, L, embed_dim)
        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, hn = self.rnn(packed)  # (B, L, embed_dim) -> (1, B, hidden_dim)
        return self.fc(hn[0])  # (1, B, hidden_dim) -> (B, num_classes)

# Bi-RNN Classifier
class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id)
        self.rnn = nn.RNN(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)  # (B, L) -> (B, L, embed_dim)
        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, hn = self.rnn(packed)  # (B, L, embed_dim) -> (2, B, hidden_dim)
        out = torch.cat((hn[0], hn[1]), dim=1)  # (2, B, hidden_dim) -> (B, hidden_dim * 2)
        return self.fc(out)  # (B, hidden_dim * 2) -> (B, num_classes)

# LSTM Classifier
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)  # (B, L) -> (B, L, embed_dim)
        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)  # (B, L, embed_dim) -> (1, B, hidden_dim)
        return self.fc(hn[0])  # (1, B, hidden_dim) -> (B, num_classes)

# Bi-LSTM Classifier
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)  # (B, L) -> (B, L, embed_dim)
        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)  # (B, L, embed_dim) -> (2, B, hidden_dim)
        out = torch.cat((hn[0], hn[1]), dim=1)  # (2, B, hidden_dim) -> (B, hidden_dim * 2)
        return self.fc(out)  # (B, hidden_dim * 2) -> (B, num_classes)


# (4) Training function
import torch.optim as optim

def train(train_loader, val_loader, model, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_accs, val_accs = [], []

    best_acc = 0.0
    best_state = None

    for epoch in range(10):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, mask)
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
            for x, mask, y in val_loader:
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                out = model(x, mask)
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

# (5) Run training and plot the train/validation accuracy
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_dim = 64
hidden_dim = 64
model = MLPClassifier(vocab_size, embed_dim, hidden_dim, num_classes).to(device)

import torchinfo
print(torchinfo.summary(
    model,
    input_data=(torch.randint(0, vocab_size, (32, 64), device=device),
                torch.ones(32, 64, dtype=torch.int64, device=device)),
    col_names=["input_size", "output_size", "num_params"]
))

train_accs, test_accs, best_state = train(train_loader, val_loader, model, device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Valid Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Train/Valid Accuracy of Model: {model.__class__.__name__} (Params: {num_params})")
plt.legend()
plt.grid(True)
plt.show()

# (6) Evaluate on test set
model.load_state_dict(best_state)
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, mask, y in test_loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        out = model(x, mask)
        all_preds.append(out.argmax(1).cpu())
        all_labels.append(y.cpu())
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
print(f"Test Accuracy: {torch.sum(all_preds == all_labels).item() / len(all_labels):.3f}")

from sklearn.metrics import classification_report
print(classification_report(all_labels.numpy(), all_preds.numpy()))
model_names = ["bert-base-uncased",
               "distilbert-base-uncased",
               "google/electra-base-discriminator"]
pretrained_model = model_names[0]

# (1) Data Loading
# Tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")

# Dataset
from torch.utils.data import RandomSampler
# MultiNLI Dataset
from datasets import load_dataset
dataset = load_dataset("multi_nli")
num_classes = len(dataset["train"].features["label"].names)
print(f"Number of classes: {num_classes}")
def tokenize(batch):
    return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, padding=False)
tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
train_set = tokenized["train"]
val_set = tokenized["validation_matched"]
test_set = tokenized["validation_mismatched"]
subset_size = len(train_set) // 30   # 1/30 of the training set is used
train_sampler = RandomSampler(train_set, num_samples=subset_size)

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
from transformers import AutoModel

# Transformer Encoder Classifier
class TransformerEncoderClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes, freeze_encoder=True):
        super(TransformerEncoderClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        # Freeze pre-trained model parameters
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, x, mask):
        outputs = self.encoder(input_ids=x, attention_mask=mask) # [batch_size, seq_len, hidden_size]
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token vector, [batch_size, hidden_size]
        logits = self.classifier(cls_output)  # [batch_size, num_classes]
        return logits


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
import torchinfo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerEncoderClassifier(pretrained_model, num_classes).to(device)

print(torchinfo.summary(
    model,
    input_data=(torch.randint(0, vocab_size, (32, 64), device=device), 
                torch.ones(32, 64, dtype=torch.int64, device=device)),
    col_names=["input_size", "output_size", "num_params"]
))

# Measure the training time
import time
start_time = time.time()
print("Starting training...")
train_accs, test_accs, best_state = train(train_loader, val_loader, model, device)
end_time = time.time()
execution_time = end_time - start_time
print(f"Training time: {execution_time:.2f} seconds")
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Valid Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Classifier with {pretrained_model} (Params: {num_params}, Time: {execution_time:.2f}s)")
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
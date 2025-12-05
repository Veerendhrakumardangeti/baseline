import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

class ECGDataset(Dataset):
    def __init__(self, X, y, transforms=None):
        if X.ndim == 2:
            X = X[:, None, :]
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transforms:
            x = self.transforms(x)
        x = np.asarray(x, dtype=np.float32)
        return torch.from_numpy(x), int(y)

class Baseline1DCNN(nn.Module):
    def __init__(self, in_channels=1, n_classes=5, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)

def add_noise(x):
    return x + np.random.normal(0, 0.01, x.shape)

def scale_signal(x):
    s = np.random.uniform(0.9, 1.1)
    return x * s

def default_transforms(x):
    x = add_noise(x)
    x = scale_signal(x)
    return x

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0
    preds = []
    trues = []
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        preds.append(out.detach().cpu().numpy())
        trues.append(y.cpu().numpy())
    preds = np.argmax(np.concatenate(preds), axis=1)
    trues = np.concatenate(trues)
    acc = (preds == trues).mean()
    return total_loss / len(loader.dataset), acc

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item() * x.size(0)
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.argmax(np.concatenate(preds), axis=1)
    trues = np.concatenate(trues)
    acc = (preds == trues).mean()
    return total_loss / len(loader.dataset), acc, trues, preds

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(path):
        return np.load(path)

    X_train = load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = load(os.path.join(args.data_dir, "y_train.npy"))
    X_val = load(os.path.join(args.data_dir, "X_val.npy"))
    y_val = load(os.path.join(args.data_dir, "y_val.npy"))
    X_test = load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = load(os.path.join(args.data_dir, "y_test.npy"))

    n_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    in_channels = 1 if X_train.ndim == 2 else X_train.shape[1]

    train_ds = ECGDataset(X_train, y_train, transforms=default_transforms if args.augment else None)
    val_ds = ECGDataset(X_val, y_val)
    test_ds = ECGDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    class_counts = np.bincount(y_train, minlength=n_classes)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    model = Baseline1DCNN(in_channels, n_classes, args.dropout).to(device)

    if args.resume:
        print(f"Loading weights from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

    best_loss = 1e9
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}/{args.epochs}  Train Loss={tr_loss:.4f} Acc={tr_acc:.4f}  Val Loss={val_loss:.4f} Acc={val_acc:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "best_model.pth"))
            print("Saved best model")

    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "best_model.pth"), map_location=device))
    test_loss, test_acc, true, pred = eval_epoch(model, test_loader, loss_fn, device)

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)
    print(confusion_matrix(true, pred))
    print(classification_report(true, pred, digits=4))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./results/baseline")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--ckpt_dir", type=str, default="./results/baseline_ckpt")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    main(args)

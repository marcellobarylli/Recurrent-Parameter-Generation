# set global seed
import random
import numpy as np
import torch
seed = SEED = 20
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

try:  # relative import
    from model import Model
    from dataset import BinaryClassifierDataset as Dataset
    from dataset import get_optimize_class
except ImportError:
    from .model import Model
    from .dataset import BinaryClassifierDataset as Dataset
    from .dataset import get_optimize_class

# import
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn import functional as F
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# load additional config
import json
config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
with open(config_file, "r") as f:
    additional_config = json.load(f)




# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    "dataset_root": "from_additional_config",
    "batch_size": 250 if __name__ == "__main__" else 50,
    "num_workers": 20,
    "pre_learning_rate": 0.01,
    "learning_rate": 3e-5,
    "pre_epochs": 2,
    "epochs": 13,
    "weight_decay": 0.1,
    "save_learning_rate": 1e-5,
    "total_save_number": 10,
    "tag": os.path.basename(os.path.dirname(__file__)),
    "optimize_class": get_optimize_class()[0],
    "optimize_class_int": get_optimize_class()[1],
}
config.update(additional_config)
print("Training:", config["optimize_class"])




# Data
dataset = Dataset(
    root=config["dataset_root"],
    train=True,
    optimize_class=config["optimize_class"],
)
train_loader = DataLoader(
    dataset=dataset,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    persistent_workers=True,
)
test_loader = DataLoader(
    dataset=Dataset(
        root=config["dataset_root"],
        train=False,
        optimize_class=config["optimize_class"],
    ),
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    shuffle=False,
)

# Model
model, head = Model()
model = model.to(device)
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
criterion = FocalLoss()

# Optimizer
head_optimizer = optim.AdamW(
    head.parameters(),
    lr=config["pre_learning_rate"],
    weight_decay=config["weight_decay"],
)
optimizer = optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["epochs"],
    eta_min=config["save_learning_rate"],
)




# Training
def train(model=model, optimizer=optimizer, scheduler=scheduler):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()

# test
@torch.no_grad()
def test(model=model):
    model.eval()
    all_targets = []
    all_predicts = []
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        # to logging losses
        all_targets.extend(targets.flatten().tolist())
        test_loss += loss.item()
        _, predicts = outputs.max(1)
        all_predicts.extend(predicts.flatten().tolist())
        total += targets.size(0)
        correct += predicts.eq(targets).sum().item()
    loss = test_loss / (batch_idx + 1)
    acc = correct / total
    print(f"Loss: {loss:.4f} | Acc: {acc:.4f}\n")
    model.train()
    return loss, acc, all_targets, all_predicts

# save train
def save_train(model=model, optimizer=optimizer):
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=min(len(dataset) // config["total_save_number"], config["batch_size"]),
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True,
    )
    model.train()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # Save checkpoint
        _, acc, _, _ = test(model=model)
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_state = {key: value.cpu().to(torch.float32) for key, value in model.state_dict().items()}
        torch.save(save_state, f"checkpoint/{str(batch_idx).zfill(4)}_acc{acc:.4f}_class{config['optimize_class_int']}_{config['tag']}.pth")
        print("save:", f"checkpoint/{str(batch_idx).zfill(4)}_acc{acc:.4f}_class{config['optimize_class_int']}_{config['tag']}.pth")
        # exit loop
        if batch_idx+1 == config["total_save_number"]:
            break




# main
if __name__ == '__main__':
    for epoch in range(config["pre_epochs"]):
        train(model=model, optimizer=head_optimizer, scheduler=None)
        test(model=model)
    for epoch in range(config["epochs"]):
        train(model=model, optimizer=optimizer, scheduler=scheduler)
        test(model=model)
    save_train(model=model, optimizer=optimizer)
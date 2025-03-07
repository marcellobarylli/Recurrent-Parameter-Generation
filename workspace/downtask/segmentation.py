import sys, os, json
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Recurrent-Parameter-Generation")+1])
sys.path.append(root)
os.chdir(root)
with open("./workspace/config.json", "r") as f:
    additional_config = json.load(f)
USE_WANDB = additional_config["use_wandb"]

# set global seed
import random
import numpy as np
import torch
seed = SEED = 1000
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# other
import _thread
import math
import random
import warnings
from _thread import start_new_thread
warnings.filterwarnings("ignore", category=UserWarning)
if USE_WANDB: import wandb
# torch
import torch
import torch.nn as nn
import bitsandbytes.optim as optim
from torch.nn import functional as F
from torch.cuda.amp import autocast
# model
from mamba_ssm import Mamba2 as Mamba
from model import MambaDiffusion as Model
from model.diffusion import DDPMSampler, DDIMSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import AutocastKwargs
from accelerate import Accelerator
# dataset
from dataset import ADE20KSegmentation as Dataset
from torch.utils.data import DataLoader




config = {
    "resume": False,
    "seed": SEED,
    # dataset setting
    "dataset": Dataset,
    "dim_per_token": 16384,
    "sequence_length": 'auto',
    # train setting
    "batch_size": 2,
    "num_workers": 4,
    "total_steps": 120000,
    "learning_rate": 0.00001,
    "weight_decay": 0.0,
    "save_every": 120000//30,
    "print_every": 50,
    "autocast": lambda i: 5000 < i < 45000,
    "checkpoint_save_path": "./checkpoint",
    # test setting
    "test_batch_size": 1,  # fixed, don't change this
    "generated_path": Dataset.generated_path,
    "test_command": Dataset.test_command,
    # to log
    "model_config": {
        "num_permutation": "auto",
        # mamba config
        "d_condition": 1,
        "d_model": 12288,
        "post_d_model": 16384,
        "d_state": 128,
        "d_conv": 4,
        "expand": 2,
        "num_layers": 2,
        # diffusion config
        "diffusion_batch": 512,
        "layer_channels": [1, 64, 96, 64, 1],
        "model_dim": 16384,
        "condition_dim": 16384,
        "kernel_size": 7,
        "sample_mode": DDIMSampler,
        "beta": (0.0001, 0.02),
        "T": 1000,
        "forward_once": True,
    },
    "tag": "downtask_segmentation",
}




# Data
print('==> Preparing data..')
train_set = config["dataset"](dim_per_token=config["dim_per_token"])
print("Dataset length:", train_set.real_length)
print("input shape:", train_set[0][0].shape)
if config["model_config"]["num_permutation"] == "auto":
    config["model_config"]["num_permutation"] = train_set.max_permutation_state
if config["model_config"]["condition_dim"] == "auto":
    config["model_config"]["condition_dim"] = config["model_config"]["d_model"]
if config["model_config"]["model_dim"] == "auto":
    config["model_config"]["model_dim"] = config["dim_per_token"]
if config["sequence_length"] == "auto":
    config["sequence_length"] = train_set.sequence_length
    print(f"sequence length: {config['sequence_length']}")
else:  # set fixed sequence_length
    assert train_set.sequence_length == config["sequence_length"], f"sequence_length={train_set.sequence_length}"
train_loader = DataLoader(
    dataset=train_set,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    persistent_workers=True,
    drop_last=True,
    shuffle=True,
)

# Model
print('==> Building model..')
Model.config = config["model_config"]
model = Model(
    sequence_length=config["sequence_length"],
    positional_embedding=train_set.get_position_embedding(
        positional_embedding_dim=config["model_config"]["d_model"]
    )  # positional_embedding
)  # model setting is in model
class VaryMambaModel(nn.Module):
    config = {}
    def __init__(self, positional_embedding):
        super().__init__()
        mamba1 = Mamba(d_model=config["model_config"]["d_model"],
                       d_state=config["model_config"]["d_state"],
                       d_conv=config["model_config"]["d_conv"],
                       expand=config["model_config"]["expand"])
        mamba2 = Mamba(d_model=config["model_config"]["post_d_model"],
                       d_state=config["model_config"]["d_state"],
                       d_conv=config["model_config"]["d_conv"],
                       expand=config["model_config"]["expand"])
        mamba2.in_proj = nn.Linear(mamba1.out_proj.out_features, mamba2.in_proj.out_features, bias=False)
        self.mamba_forward = nn.Sequential(*[mamba1, mamba2])
        pe = positional_embedding[None, :, :]
        if self.config.get("trainable_pe"):
            self.pe = nn.Parameter(pe)
        else:  # fixed positional embedding
            self.register_buffer("pe", pe)
    def forward(self, output_shape, condition=None):
        x = self.mamba_forward(self.pe.repeat(output_shape[0], 1, 1) + condition)
        return x
VaryMambaModel.config = config["model_config"]
model.model = VaryMambaModel(
    positional_embedding=train_set.get_position_embedding(
        positional_embedding_dim=config["model_config"]["d_model"]
    )  # positional_embedding
)  # update mamba model
torch.cuda.empty_cache()

# Optimizer
print('==> Building optimizer..')
optimizer = optim.AdamW8bit(
    params=model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)
scheduler = CosineAnnealingLR(
    optimizer=optimizer,
    T_max=config["total_steps"],
)

# load checkpoint
if config["resume"] and os.path.exists(f"./cache_{config['tag']}.pt"):
    diction = torch.load(f"./cache_{config['tag']}.pt", map_location="cpu")
    model.load_state_dict(diction["model"])
    optimizer.load_state_dict(diction["optimizer"])
    scheduler.load_state_dict(diction["scheduler"])
    start_batch_idx = diction["step"] + 1
else:  # not resume
    start_batch_idx = 0

# accelerator
if __name__ == "__main__":
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs,])
    if config["dim_per_token"] > 12288 and accelerator.state.num_processes == 1:
        print(f"\033[91mWARNING: With token size {config['dim_per_token']}, we suggest to train on multiple GPUs.\033[0m")
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)


# wandb
if __name__ == "__main__" and USE_WANDB and accelerator.is_main_process:
    wandb.login(key=additional_config["wandb_api_key"])
    wandb.init(project="Recurrent-Parameter-Generation", name=config['tag'], config=config,)




# Training
print('==> Defining training..')
def train():
    if not USE_WANDB:
        train_loss = 0
        this_steps = 0
    print("==> Start training..")
    model.train()
    for batch_idx, (param, permutation_state) in enumerate(train_loader):
        batch_idx += start_batch_idx
        optimizer.zero_grad()
        # train
        # noinspection PyArgumentList
        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=config["autocast"](batch_idx))):
            loss = model(output_shape=param.shape, x_0=param, permutation_state=permutation_state)
        accelerator.backward(loss)
        optimizer.step()
        if accelerator.is_main_process:
            scheduler.step()
        # to logging losses and print and save
        if USE_WANDB and accelerator.is_main_process:
            wandb.log({"train_loss": loss.item()})
        elif USE_WANDB:
            pass  # don't print
        else:  # not use wandb
            train_loss += loss.item()
            this_steps += 1
            if this_steps % config["print_every"] == 0:
                print('Loss: %.6f' % (train_loss/this_steps))
                this_steps = 0
                train_loss = 0
        if batch_idx % config["save_every"] == 0 and accelerator.is_main_process:
            os.makedirs(config["checkpoint_save_path"], exist_ok=True)
            state = accelerator.unwrap_model(model).state_dict()
            torch.save(state, os.path.join(config["checkpoint_save_path"], config["tag"]+".pth"))
            torch.save({
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": accelerator.unwrap_model(optimizer).state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": batch_idx
            }, f"./cache_{config['tag']}.pt")
            generate(save_path=config["generated_path"], need_test=True)
        if batch_idx >= config["total_steps"]:
            break


def generate(save_path=config["generated_path"], need_test=True):
    print("\n==> Generating..")
    model.eval()
    with torch.no_grad():
        prediction = model(sample=True)
        generated_norm = prediction.abs().mean()
    print("Generated_norm:", generated_norm.item())
    if USE_WANDB:
        wandb.log({"generated_norm": generated_norm.item()})
    train_set.save_params(prediction, save_path=save_path)
    if need_test:
        _thread.start_new_thread(os.system, (config["test_command"],))  # not stuck here
    model.train()
    return prediction




if __name__ == '__main__':
    train()
    del train_loader  # deal problems by dataloader
    print("Finished Training!")
    exit(0)
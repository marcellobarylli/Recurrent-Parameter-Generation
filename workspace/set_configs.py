import os
import os.path as op
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Recurrent-Parameter-Generation")+1])




assert op.exists(root), "Cannot find the executing root."
assert op.basename(root) == "Recurrent-Parameter-Generation", \
    f"""
    You need to rename the repository folder to "Recurrent-Parameter-Generation" manually.
    Because the whole project depends on this name.
    The file structure is as follow:
    └─Recurrent-Parameter-Generation
        ├─dataset
        │   ├─cifar10_cnnmedium
        │   ├─...(total 21 folders)
        │   ├─__init__.py
        │   ├─config.json
        │   ├─dataset.py
        │   └─register.py
        ├─model
        │   ├─__init__.py
        │   ├─denoiser.py
        │   ├─diffusion.py
        │   └─...(total 8 files)
        ├─quick_start
        │   ├─set_configs.py
        │   └─auto_start.sh
        ├─workspace
        │   ├─main
        │   ├─evaluate
        │   ├─...(total 6 folders)
        │   └─config.json
        ├─README.md
        └─requirements.txt
    """


print("\n1. Set an \033[91mABSOLUTE\033[0m path to download your small dataset, such as CIFAR10 and CIFAR100")
default_dataset_root = op.join(op.dirname(op.abspath(root)), 'Dataset')
dataset_root = input(f"[{default_dataset_root} (default & \033[32mrecommanded\033[0m)]: ") or default_dataset_root
print(f"\033[32mdataset_root is set to {dataset_root}\033[0m")


print("\n2. Set the \033[91mABSOLUTE\033[0m path to your ImageNet1k dataset. "
      "\033[32m(Press ENTER if you don't want to use ImageNet1k)\033[0m")
print("""The ImageNet1k dataset should be organized as follow:
└─ImageNet1k
   ├─train
   │   ├─n01443537
   │   ├─n01484850
   │   ├─n########
   └─test
       ├─n01443537
       ├─n01484850
       └─n########""")
imagenet_root = input(f"[None (default)]: ")
if imagenet_root == "":
    print("\033[32mWe don't use ImageNet1k.\033[0m")
    imagenet_root_train = None
    imagenet_root_test = None
else:  # imagenet path is set
    print(f"\033[32mimagenet_root is set to {imagenet_root}\033[0m")
    imagenet_root_train = op.join(imagenet_root, "train")
    imagenet_root_test = op.join(imagenet_root, "test")
    assert op.exists(imagenet_root_train), f"{imagenet_root_train} is not existed."
    assert op.exists(imagenet_root_test), f"{imagenet_root_test} is not existed."


print("\n3. Do you want to use wandb?")
default_use_wandb = True
use_wandb = input("[True (default & \033[32mrecommanded\033[0m)) / False]: ")
use_wandb = default_use_wandb if use_wandb == "" else eval(use_wandb)
print(f"\033[32muse_wandb is set to {use_wandb}\033[0m")
if use_wandb:
    wandb_api_key = input("Set your wandb api key: ")
    assert wandb_api_key != "", "You need to set an API_KEY is you want to use wandb."




print()
import json
from pprint import pprint

# dataset/config.json
print()
with open(op.join(root, "dataset/config.json"), "r") as f:
    dataset_config = json.load(f)
    dataset_config.update({
        "dataset_root": dataset_root,
        "imagenet_root": {
            "train": imagenet_root_train,
            "test": imagenet_root_test,
        },
    })
with open(op.join(root, "dataset/config.json"), "w") as f:
    print("\033[32mUpdated dataset/config.json is as follow:\033[0m")
    pprint(dataset_config)
    json.dump(dataset_config, f)

# workspace/config.json
print()
with open(op.join(root, "workspace/config.json"), "r") as f:
    workspace_config = json.load(f)
    workspace_config.update({
        "use_wandb": use_wandb,
        "wandb_api_key": wandb_api_key,
    })
with open(op.join(root, "workspace/config.json"), "w") as f:
    print("\033[32mUpdated workspace/config.json is as follow:\033[0m")
    pprint(workspace_config)
    json.dump(workspace_config, f)
print()
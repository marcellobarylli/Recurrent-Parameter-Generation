import os
import re
import sys
import time
import torch
import shutil
from _thread import start_new_thread
cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
assert len(cuda_visible_devices) == 1, "Only support train on one GPU."
RANK = 4




checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint")
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=False)
generated_path = os.path.join(os.path.dirname(__file__), "generated")
if not os.path.exists(generated_path):
    os.makedirs(generated_path, exist_ok=False)
adapter_config_path = os.path.join(os.path.dirname(__file__), "adapter_config.json")


# change root
import json
config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
with open(config_file, "r") as f:
    additional_config = json.load(f)
root = additional_config["dora_root"]
sys.path.append(root)
os.chdir(root)
print(f"\033[91mWe are working under: {root}\033[0m")
if os.path.exists(f"./finetuned_result/dora_r{RANK}"):
    print(f"\033[91mWARNING: ./finetuned_result/dora_r{RANK} existed!\033[0m")
    input("\033[91mPress ENTER to clear this dir...\033[0m")
    os.system(f"rm ./finetuned_result/dora_r{RANK}/* -rf")




exit_flag = False

def move_to_checkpoint():
    global exit_flag
    index = 1
    finished_list = []
    while exit_flag is False:
        father = f"./finetuned_result/dora_r{RANK}"
        if not os.path.exists(father):
            time.sleep(1)
            continue
        item_list = os.listdir(father)
        for item in item_list:
            src = os.path.join(father, item)
            if not os.path.isdir(src):
                continue  # is file saved in the end
            if item[:4] == "tmp-":
                continue  # is a tmp file
            if src in finished_list:
                continue  # have been processed
            finished_list.append(src)
            try:  # deleted before loaded
                shutil.copy(os.path.join(src, "adapter_config.json"), adapter_config_path)
                src = os.path.join(src, "adapter_model.bin")
                diction = torch.load(src, map_location="cpu", weights_only=False)
                dst = os.path.join(checkpoint_path, f"{str(index).zfill(7)}.pth")
                torch.save(diction, dst)
            except Exception as e:
                print(f"\033[91mWARNING: encountered {e} and ignored.\033[0m")
                continue
            print(f"Moved {src} to {dst}.")
            index += 1
        time.sleep(1)
start_new_thread(move_to_checkpoint, ())


def remove_early_checkpoint():
    global exit_flag
    while exit_flag is False:
        item_list = [item for item in os.listdir(checkpoint_path) if item.endswith('.pth')]
        if len(item_list) <= 50:
            time.sleep(10)
            continue
        def extract_number(filename):
            match = re.search(r'(\d+).pth', filename)
            return int(match.group(1)) if match else -1
        sorted_items = sorted(item_list, key=extract_number)
        num_to_remove = len(sorted_items) - 50
        for i in range(num_to_remove):
            file_to_remove = os.path.join(checkpoint_path, sorted_items[i])
            os.remove(file_to_remove)
            print(f"\033[91mRemoved: {file_to_remove}\033[0m")
        time.sleep(10)
start_new_thread(remove_early_checkpoint, ())




# start training
activate_path = shutil.which('conda')[:-5] + "activate"
env_path = shutil.which('conda')[:-9] + f"envs/{additional_config['dora_env_name']}"
os.system(
    f"bash -c \"source {activate_path} {env_path} && " +
    f"sh llama_7B_Dora.sh {RANK} {RANK*2} ./finetuned_result/dora_r{RANK} {cuda_visible_devices}\""
)
# noinspection PyRedeclaration
time.sleep(5)
exit_flag = True
time.sleep(20)
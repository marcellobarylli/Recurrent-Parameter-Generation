import os
import re
import sys
import time
import shutil
from _thread import start_new_thread
cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
assert len(cuda_visible_devices) == 1, "Only support test on one GPU."
RANK = 16




checkpoint_path = sys.argv[1]
assert os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path)
checkpoint_path = os.path.abspath(checkpoint_path)
adapter_config_path = os.path.join(os.path.dirname(__file__), "adapter_config.json")
assert os.path.exists(adapter_config_path)
evaluate_path = f"./finetuned_result/dora_r{RANK}/evaluating"


# change root
import json
config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
with open(config_file, "r") as f:
    additional_config = json.load(f)
root = additional_config["dora_root"]
sys.path.append(root)
os.chdir(root)
print(f"\033[91mWe are working under: {root}\033[0m")


# copy file
os.makedirs(evaluate_path, exist_ok=True)
if len(os.listdir(evaluate_path)) != 0:
    os.system(f"rm {evaluate_path}/* -rf")
shutil.copy(adapter_config_path, os.path.join(evaluate_path, "adapter_config.json"))
shutil.copy(checkpoint_path, os.path.join(evaluate_path, "adapter_model.bin"))
evaluate_path = os.path.abspath(evaluate_path)




def find_last_accuracy_value_in_result_txt(file_pointer):
    original_position = file_pointer.tell()
    file_pointer.seek(0, 2)
    end_position = file_pointer.tell()
    buffer_size = 1024
    while end_position > 0:
        start_position = max(0, end_position - buffer_size)
        file_pointer.seek(start_position)
        chunk = file_pointer.read(end_position - start_position)
        lines = chunk.splitlines(True)  # True keeps the newline character with the line
        if start_position > 0:
            first_line_in_chunk = lines[0]
            lines[0] = file_pointer.readline() + first_line_in_chunk
        for line in reversed(lines):
            match = re.search(r"accuracy\s\d+\s+(\d+\.\d+)", line)
            if match:
                accuracy_value = float(match.group(1))
                file_pointer.seek(original_position)
                return accuracy_value
        end_position = start_position
    file_pointer.seek(original_position)
    return None


def conclude(path=evaluate_path):
    print("\n\n\n\n\n==================== CONCLUDE ======================\n")
    files = os.listdir(path)
    files.sort()
    for file in files:
        file = os.path.join(path, file)
        if ".txt" in file:
            name = os.path.basename(file).split(".")[0]
            with open(file, "r") as f:
                value = find_last_accuracy_value_in_result_txt(f)
            value *= 100.
            print(f"{name}: {value:.3f}")




# start testing
activate_path = shutil.which('conda')[:-5] + "activate"
env_path = shutil.which('conda')[:-9] + f"envs/{additional_config['dora_env_name']}"
os.system(
    f"bash -c \"source {activate_path} {env_path} && " +
    f"sh llama_7B_Dora_eval.sh {evaluate_path} {cuda_visible_devices}\""
)
conclude(evaluate_path)
print()

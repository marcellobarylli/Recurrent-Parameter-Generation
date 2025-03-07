import sys, os, json
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Recurrent-Parameter-Generation")+1])
sys.path.append(root)
os.chdir(root)

# torch
import torch
# father
import importlib
item = importlib.import_module(f"{sys.argv[1]}")
Dataset = item.Dataset
train_set = item.train_set
config = item.config
model = item.model
vae = item.vae
assert config.get("tag") is not None, "Remember to set a tag."




generate_config = {
    "device": "cuda",
    "num_generated": 200,
    "checkpoint": f"./checkpoint/{config['tag']}.pth",
    "generated_path": os.path.join(Dataset.generated_path.rsplit("/", 1)[0], "generated_{}_{}.pth"),
    "test_command": os.path.join(Dataset.test_command.rsplit("/", 1)[0], "generated_{}_{}.pth"),
    "need_test": True,
}
config.update(generate_config)




# Model
print('==> Building model..')
diction = torch.load(config["checkpoint"], map_location="cpu")
vae.load_state_dict(diction["vae"])
model.load_state_dict(diction["diffusion"])
model = model.to(config["device"])
vae = vae.to(config["device"])


# generate
print('==> Defining generate..')
def generate(save_path=config["generated_path"], test_command=config["test_command"], need_test=True):
    print("\n==> Generating..")
    model.eval()
    with torch.cuda.amp.autocast(True, torch.bfloat16):
        with torch.no_grad():
            mu = model(sample=True)
            mu = torch.randn_like(mu)
            prediction = vae.decode(mu)
            generated_norm = torch.nanmean(prediction.abs())
    print("Generated_norm:", generated_norm.item())
    train_set.save_params(prediction, save_path=save_path)
    if need_test:
        os.system(test_command)




if __name__ == "__main__":
    for i in range(config["num_generated"]):
        index = str(i+1).zfill(3)
        print("Save to", config["generated_path"].format(config["tag"], index))
        generate(
            save_path=config["generated_path"].format(config["tag"], index),
            test_command=config["test_command"].format(config["tag"], index),
            need_test=config["need_test"],
        )
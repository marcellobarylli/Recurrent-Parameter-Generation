



## Environment
1. Create your conda environment.
```
conda create -n rpg python=3.11
conda activate rpg
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
2. Install mamba-ssm. (You may run into compilation issues, refer to the [official mamba-ssm repository](https://github.com/state-spaces/mamba) for details.)
```
pip install mamba-ssm[causal-conv1d]
pip install causal-conv1d
```
3. Install other dependencies for this repository.
```
git clone https://github.com/NUS-HPC-AI-Lab/Recurrent-Parameter-Generation.git
cd Recurrent-Parameter-Generation
pip install -r requirements.txt
```

## Workflow
1. Modify your config file.
```
# Set up your configs interactively.
python ./quick_start/check_configs.py
```

2. Prepare checkpoint datasets.
```
cd ./dataset/cifar10_cnnmedium
CUDA_VISIBLE_DEVICES=0 python train.py
cd ../..
```

3. Train RPG model.
```
sh ./quick_start/resnet18_example.sh 0
# '0' in the end means gpu_ids=0
```

4. Generate and test.
```
cd ./workspace/evaluate
CUDA_VISIBLE_DEVICES=0 python generate.py quick_start.resnet18_example
cd ../..
```


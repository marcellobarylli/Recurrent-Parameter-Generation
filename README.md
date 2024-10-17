



## Environment
Before you get started, you need to set up a conda environment first.
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




## Quick Start
This section covers the entire process from preparing the checkpoint dataset to training and testing the RPG model.

1. Modify your config file.
```
# Set up your configs interactively.
python ./quick_start/setting_configs.py
```

2. Prepare checkpoint datasets.
```
cd ./dataset/cifar10_cnnmedium
CUDA_VISIBLE_DEVICES=0 python train.py
cd ../..
```

3. Train RPG model. ('0' refers to GPU index)
```
cd quick_start
sh resnet18_example.sh '0'
cd ..
```

4. Generate and test.
```
CUDA_VISIBLE_DEVICES=0 python ./workspace/evaluate/generate.py quick_start.resnet18_example
```

More examples and experiments are located in the `workspace` directory, 
written with a structure similar to that of `quick_start`.




## Advanced Usage
In this section, we will cover how to reproduce the experiments from [Section-4]() of our paper, how to adapt other downstream tasks (using DORA on LLaMA as an example), and how to adapt your own checkpoint dataset into our codes.


<details>
  <summary style="font-size:1.3em; font-weight:bold;">Reproduce Section-4</summary>

In this section, we will guide you step by step through reproducing the experiments from Section-4. Similar to Quick Start section. You can execute the following commands.

1. Modify your config file. (You can skip this step if you have done.)
```
python ./quick_start/setting_configs.py
```

2. Prepare checkpoint dataset. (Choose one of two options.)
```
# Download our dataset. (download about 64 GB)
cd ./dataset/condition_classinput_vittiny
wget https://xxxx/xxxx/xxxx/HPCAI.tar
tar -xvf HPCAI.tar
rm HPCAI.tar
cd ../..
```
```
# Train from the codes. (need a really long time)
cd ./dataset/condition_classinput_vittiny
CUDA_VISIBLE_DEVICES=0 bash train.sh
sh split.sh
cd ../..
```

3. Train RPG model. ('1,2,3,4' refers to GPU index)
```
cd ./workspace/condition 
sh generalization.sh '1,2,3,4'
cd ../..
```

4. Generate and test.
```
# Generate parameters for 20 random seen tasks
CUDA_VISIBLE_DEVICES=0 python ./workspace/condition/generate_seen.py

# Generate parameters for all unseen tasks
CUDA_VISIBLE_DEVICES=0 python ./workspace/condition/generate_unseen.py
```

5. Check more detailed results.
```
cd ./dataset/condition_classinput_vittiny
CUDA_VISIBLE_DEVICES=0 python detail.py ./generated/generated_generalization_class0279.pth
cd ../..
```

</details>


<details>
  <summary style="font-size:1.3em; font-weight:bold;">Adapt other downstream tasks</summary>

In this section, we will use the DoRA adaptor for Llama as an example to teach you how to apply our method to more complex downstream tasks.

1. Create another conda environment for dora_llama following the [official repositories](https://github.com/NVlabs/DoRA), especially for [commonsense reasoning](https://github.com/NVlabs/DoRA/tree/main/commonsense_reasoning).
Meanwhile, you need to clone the repositories for dora_llama to any path you like. Then you should get a directory structure like this ("..." means there are many other files or folders here, but those are not important to us.):
```
└─DoRA
   ├─commonsense_reasoning
   │  ├─dataset
   │  │  ├─ARC-Challenge
   │  │  ├─ARC-Easy
   │  │  ├─boolq
   │  │  ├─hellaswag
   │  │  ├─openbookqa
   │  │  ├─piqa
   │  │  ├─social_i_qa
   │  │  ├─winogrande
   │  │  └─...
   │  ├─peft
   │  │  ├─src
   │  │  │  └─peft
   │  │  └─...
   │  ├─commonsense_170k.json
   │  ├─commonsense_evaluate.py
   │  ├─finetune.py
   │  ├─llama_7B_Dora.sh
   │  └─llama_7B_Dora_eval.sh
   └─...
```
- You could try the official finetuning and testing code of dora_llama under `/path/to/your/DoRA/commonsense_reasoning`. If everything is working properly, it means all of your operation is correct.
(Since waiting for the finetuning takes a lot of time, you can skip this step first. If there are any problems in the future, you can come back to test your dora_llama configuration.)
```
# execute under /path/to/your/DoRA/commonsense_reasoning

# finetuning
sh llama_7B_Dora.sh 32 64 ./finetuned_result/dora_r32 0

# testing
sh llama_7B_Dora_eval.sh ./finetuned_result/dora_r32 0
```


2. Modify your config file.
```
vim ./dataset/config.json

###################### content in config.json ######################
{
  ...
  "dora_root": "/ABSOLUTE/path/to/your/DoRA/commonsense_reasoning",
  "dora_env_name": "dora_llama"
}
###################### content in config.json ######################
```

3. Prepare checkpoint dataset.
```
cd ./dataset/downtask_dora_r4
CUDA_VISIBLE_DEVICES=0 python train.py
cd ../..
```

4. Train RPG model. ('0' refers to GPU index)
```
cd ./workspace/downtask
sh dora_r4.sh '0'
cd ../..
```

5. Generate and test. (We recommend separating the generation and testing processes because the testing process is complex and time-consuming, and the separated operation makes it easier to check the results.)
```
# Generate without testing
CUDA_VISIBLE_DEVICES=0 python ./workspace/evaluate/generate.py workspace.downtask.dora_r4 "need_test=False,num_generated=5"

# Test one by one manually.
cd ./dataset/downtask_dora_r4
CUDA_VISIBLE_DEVICES=0 python test.py ./generated/generated_downtask_dora_r4_001.pth
CUDA_VISIBLE_DEVICES=0 python test.py ./generated/generated_downtask_dora_r4_002.pth
CUDA_VISIBLE_DEVICES=0 python test.py ./generated/generated_downtask_dora_r4_003.pth
CUDA_VISIBLE_DEVICES=0 python test.py ./generated/generated_downtask_dora_r4_004.pth
CUDA_VISIBLE_DEVICES=0 python test.py ./generated/generated_downtask_dora_r4_005.pth
cd ../..
```

Please note that the methods mentioned above involve automatically activating a specified conda environment through the `dataset/downtask_dora_r4/test.py` and `dataset/downtask_dora_r4/train.py` files, and executing the official training and testing shell script of dora_llama. 
For more details, you can check the specific contents of `dataset/downtask_dora_r4/test.py` and `dataset/downtask_dora_r4/train.py`.

</details>


<details>
  <summary style="font-size:1.3em; font-weight:bold;">Adapt your own dataset</summary>

In this section, we will introduce how to register your own model in this code framework and use RPG to generate its parameters.

1. Create a dataset
```
cd ./dataset
mkdir your_dataset_name
cd ./your_dataset_name
```
- In this directory, there are three necessary items. 
  1. A checkpoint folder is used to store checkpoints used for training. All the pretrained checkpoints should be placed in this folder.
  2. A generated folder is used to store the checkpoints to be generated. Before you start training RPG models, this folder should be empty.
  3. A test.py is used to test the specified checkpoint and output the test results. This test.py accepts a CLI argument that is the path to the checkpoint to be tested, which will facilitate subsequent calls.
- The structure should be as following ("..." means there are many other files or folders here, but those are not important.):
```
└─Recurrent-Parameter-Generation
   ├─dataset
   │  ├─your_dataset_name
   │  │  ├─checkpoint
   │  │  │  ├─your_1st_checkpoint_name.pth
   │  │  │  ├─your_2nd_checkpoint_name.pth
   │  │  │  ├─your_3rd_checkpoint_name.pth
   │  │  │  └─...
   │  │  ├─generated
   │  │  ├─test.py
   │  │  └─...
   │  └─...
   └─...
```

- Make sure you can run this command properly.
```
# execute under /path/to/Recurrent-Parameter-Generation/dataset/your_dataset_name
python test.py ./checkpoint/your_1st_checkpoint_name.pth
```

2. Register your dataset. You need to write your own dataset class in the `dataset/register.py` file, which contains three class variables.
```
# execute under /path/to/Recurrent-Parameter-Generation
vim ./dataset/register.py

####################################### add to the end of register.py #######################################
class YourDatasetName(BaseDataset):
    data_path = "./dataset/your_dataset_name/checkpoint"
    generated_path = "./dataset/your_dataset_name/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/your_dataset_name/test.py " + \
                   "./dataset/your_dataset_name/generated/generated_model.pth"
####################################### add to the end of register.py #######################################
```

3. Create your training script. ('your_training_tag' is decided by yourself. And you can get more information for modifying the config from the appendix of [our paper]())
```diff
cp ./quick_start/resnet18_example.py ./workspace/your_training_tag.py
vim ./workspace/your_training_tag.py

######################## on line 43 in your_training_tag.py ########################
- from dataset import Cifar10_ResNet18 as Dataset
+ from dataset import YourDatasetName as Dataset
######################## on line 43 in your_training_tag.py ########################

###################### on line 49-91 in your_training_tag.py #######################
  config = {
      "seed": SEED,
      # dataset setting
      "dataset": Dataset,
-     "dim_per_token": 8192,
+     "dim_per_token": suitable_token_size_for_you,
      "sequence_length": 'auto',
      # train setting
-     "batch_size": 8,
+     "batch_size": suitable_batch_size_for_you,
      "num_workers": 16,
-     "total_steps": 80000,
+     "total_steps": the_number_of_steps_you_want_to_train,
-     "learning_rate": 0.00003,
+     "learning_rate": suitable_learning_rate_for_you,
      "weight_decay": 0.0,
-     "save_every": 80000//30,
+     "save_every": number_of_interval_steps_for_saving_and_testing,
      "print_every": 50,
      "autocast": lambda i: 5000 < i < 45000,
      "checkpoint_save_path": "./checkpoint",
      # test setting
      "test_batch_size": 1,  # fixed, don't change this
      "generated_path": Dataset.generated_path,
      "test_command": Dataset.test_command,
      # to log
      "model_config": {
          "num_permutation": 'auto',
          # mamba config
          "d_condition": 1,
-         "d_model": 8192,
+         "d_model": suitable_token_size_for_you,
          "d_state": 128,
          "d_conv": 4,
          "expand": 2,
          "num_layers": 2,
          # diffusion config
-         "diffusion_batch": 512,
+         "diffusion_batch": suitable_diffusion_batch_for_you,
-         "layer_channels": [1, 32, 64, 128, 64, 32, 1],
+         "layer_channels": suitable_layer_channels_for_you,
          "model_dim": "auto",
          "condition_dim": "auto",
          "kernel_size": 7,
          "sample_mode": DDPMSampler,
          "beta": (0.0001, 0.02),
          "T": 1000,
          "forward_once": True,
      },
-     "tag": "quick_start_cifar10_resnet18",
+     "tag": "your_training_tag",
  }
###################### on line 49-91 in your_training_tag.py #######################
```
- Then your need a start shell script.
```diff
cp ./quick_start/resnet18_example.sh ./workspace/your_training_tag.sh
vim ./workspace/your_training_tag.sh

###################### content in your_training_tag.sh #######################
  accelerate launch \
    --main_process_port=0 \
    --num_processes=1 \
    --gpu_ids="$1" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
-   resnet18_example.py
+   your_training_tag.py
###################### content in your_training_tag.sh #######################
```

4. Train RPG model. ('0' refers to GPU index)
```
cd ./workspace
sh your_training_tag.sh 0
cd ..
```

4. Generate and test.
```
CUDA_VISIBLE_DEVICES=0 python ./evaluate/generate.py workspace.your_training_tag
```

</details>

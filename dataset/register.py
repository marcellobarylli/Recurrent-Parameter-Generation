import torch
from .dataset import BaseDataset, ConditionalDataset




class ImageNet_ResNet18(BaseDataset):
    data_path = "./dataset/imagenet_resnet18/checkpoint"
    generated_path = "./dataset/imagenet_resnet18/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_resnet18/test.py " + \
                   "./dataset/imagenet_resnet18/generated/generated_model.pth"

class ImageNet_ResNet50(BaseDataset):
    data_path = "./dataset/imagenet_resnet50/checkpoint"
    generated_path = "./dataset/imagenet_resnet50/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_resnet50/test.py " + \
                   "./dataset/imagenet_resnet50/generated/generated_model.pth"

class ImageNet_ViTTiny(BaseDataset):
    data_path = "./dataset/imagenet_vittiny/checkpoint"
    generated_path = "./dataset/imagenet_vittiny/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_vittiny/test.py " + \
                   "./dataset/imagenet_vittiny/generated/generated_model.pth"

class ImageNet_ViTSmall(BaseDataset):
    data_path = "./dataset/imagenet_vitsmall/checkpoint"
    generated_path = "./dataset/imagenet_vitsmall/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_vitsmall/test.py " + \
                   "./dataset/imagenet_vitsmall/generated/generated_model.pth"

class ImageNet_ViTBase(BaseDataset):
    data_path = "./dataset/imagenet_vitbase/checkpoint"
    generated_path = "./dataset/imagenet_vitbase/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_vitbase/test.py " + \
                   "./dataset/imagenet_vitbase/generated/generated_model.pth"

class ImageNet_ConvNextAtto(BaseDataset):
    data_path = "./dataset/imagenet_convnextatto/checkpoint"
    generated_path = "./dataset/imagenet_convnextatto/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_convnextatto/test.py " + \
                   "./dataset/imagenet_convnextatto/generated/generated_model.pth"

class ImageNet_ConvNextLarge(BaseDataset):
    data_path = "./dataset/imagenet_convnextlarge/checkpoint"
    generated_path = "./dataset/imagenet_convnextlarge/generated/generated_model.pth"
    test_command = "python ./dataset/imagenet_convnextlarge/test.py " + \
                   "./dataset/imagenet_convnextlarge/generated/generated_model.pth"

class CocoDetection(BaseDataset):
    data_path = "./dataset/downtask_detection/checkpoint"
    generated_path = "./dataset/downtask_detection/generated/generated_model.pth"
    test_command = "echo \"Code for testing is coming soon!\n\""
    # test_command = "bash ./dataset/downtask_detection/test.sh " + \
    #                "./dataset/downtask_detection/generated/generated_model.pth"

class ADE20KSegmentation(BaseDataset):
    data_path = "./dataset/downtask_segmentation/checkpoint"
    generated_path = "./dataset/downtask_segmentation/generated/generated_model.pth"
    test_command = "echo \"Code for testing is coming soon!\n\""
    # test_command = "bash ./dataset/downtask_segmentation/test.sh " + \
    #                "./dataset/downtask_segmentation/generated/generated_model.pth"

class DoRACommonSenseReasoningR4(BaseDataset):
    data_path = "./dataset/downtask_dora_r4/checkpoint"
    generated_path = "./dataset/downtask_dora_r4/generated/generated_model.pth"
    test_command = "echo \"Code for testing is coming soon!\n\""
    # test_command = "bash ./dataset/downtask_dora_r4/test.sh " + \
    #                "./dataset/downtask_dora_r4/generated/generated_model.pth"

class DoRACommonSenseReasoningR16(BaseDataset):
    data_path = "./dataset/downtask_dora_r16/checkpoint"
    generated_path = "./dataset/downtask_dora_r16/generated/generated_model.pth"
    test_command = "echo \"Code for testing is coming soon!\n\""
    # test_command = "bash ./dataset/downtask_dora_r16/test.sh " + \
    #                "./dataset/downtask_dora_r16/generated/generated_model.pth"

class Cifar10_ResNet18(BaseDataset):
    data_path = "./dataset/cifar10_resnet18/checkpoint"
    generated_path = "./dataset/cifar10_resnet18/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_resnet18/test.py " + \
                   "./dataset/cifar10_resnet18/generated/generated_model.pth"

class Cifar10_MobileNetv3(BaseDataset):
    data_path = "./dataset/cifar10_mobilenetv3/checkpoint"
    generated_path = "./dataset/cifar10_mobilenetv3/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_mobilenetv3/test.py " + \
                   "./dataset/cifar10_mobilenetv3/generated/generated_model.pth"

class Cifar10_ViTBase(BaseDataset):
    data_path = "./dataset/cifar10_vitbase/checkpoint"
    generated_path = "./dataset/cifar10_vitbase/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_vitbase/test.py " + \
                   "./dataset/cifar10_vitbase/generated/generated_model.pth"

class Cifar10_CNNSmall(BaseDataset):
    data_path = "./dataset/cifar10_cnnsmall/checkpoint"
    generated_path = "./dataset/cifar10_cnnsmall/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_cnnsmall/test.py " + \
                   "./dataset/cifar10_cnnsmall/generated/generated_model.pth"

class Cifar10_CNNMedium(BaseDataset):
    data_path = "./dataset/cifar10_cnnmedium/checkpoint"
    generated_path = "./dataset/cifar10_cnnmedium/generated/generated_model.pth"
    test_command = "python ./dataset/cifar10_cnnmedium/test.py " + \
                   "./dataset/cifar10_cnnmedium/generated/generated_model.pth"

class Cifar100_ResNet18BN(BaseDataset):
    data_path = "./dataset/cifar100_resnet18bn/checkpoint"
    generated_path = "./dataset/cifar100_resnet18bn/generated/generated_model.pth"
    test_command = "python ./dataset/cifar100_resnet18bn/test.py " + \
                   "./dataset/cifar100_resnet18bn/generated/generated_model.pth"




class Permutation_ViTTiny(ConditionalDataset):
    data_path = "./dataset/condition_permutation_vittiny/checkpoint"
    generated_path = "./dataset/condition_permutation_vittiny/generated/generated_model.pth"
    test_command = "python ./dataset/condition_permutation_vittiny/test.py " + \
                   "./dataset/condition_permutation_vittiny/generated/generated_model.pth"

    def _extract_condition(self, index: int):
        condition = super()._extract_condition(index)[2][5:]
        return int(condition)




class ClassInput_ViTTiny(ConditionalDataset):
    def _extract_condition(self, index: int):
        condition = super()._extract_condition(index)[2][5:]
        one_hot_string = bin(int(condition))[2:].zfill(10)
        optimize_class = [index for index, i in enumerate(one_hot_string) if i == "1"]
        indicator_tensor = torch.zeros(size=(10,))
        for i in optimize_class:
            indicator_tensor[i] = 1.0
        return indicator_tensor

class ClassInput_ViTTiny_Train(ClassInput_ViTTiny):
    data_path = "./dataset/condition_classinput_vittiny/checkpoint_train"
    generated_path = None
    test_command = None

class ClassInput_ViTTiny_Test(ClassInput_ViTTiny):
    data_path = "./dataset/condition_classinput_vittiny/checkpoint_test"
    generated_path = "./dataset/condition_classinput_vittiny/generated/generated_model_class{}.pth"
    test_command = "python ./dataset/condition_classinput_vittiny/test.py " + \
                   "./dataset/condition_classinput_vittiny/generated/generated_model_class{}.pth"

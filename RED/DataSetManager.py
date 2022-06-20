import fiftyone as fo
import fiftyone.zoo as foz
import torch
import torchvision

dataset = fo.get_default_dataset_dir("tableDataset")
session = fo.launch_app(dataset)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



import fiftyone as fo
import fiftyone.zoo as foz
import torch
import torchvision

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
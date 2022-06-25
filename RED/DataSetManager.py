import fiftyone as fo
import fiftyone.zoo as foz
import torch
import torchvision
from torchvision import *
from torch import *
import matplotlib.pyplot as plt
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# path del directorio de las imagenes
data_dir = os.path.abspath("tableDataset")


# Variable de transformación para el rescale y centering de las imagenes
transform = transforms.Compose(
    [transforms.Resize(255), transforms.CenterCrop(255), transforms.ToTensor()])

# Se carga las imagenes en la variable dataset

dataset = datasets.ImageFolder(data_dir, transform=transform)

# Loader de las imagenes
batch_size = 1295
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=True)

images, labels = next(iter(dataloader))

print('Numero de muestras : ', len(images))
print(labels)

#Metodo para mostrar algunas imagenes aleatorias
def imshow(img):
    img = img / 2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# cargamos las imagenes randoms

dataiter = iter(dataset)
images, labels = dataiter.__next__()

imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

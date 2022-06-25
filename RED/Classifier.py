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

# Variable de transformacion para las imagenes cargadas
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Cantidad de imagenes que se van a cargar para entrenar el clasificador
batch_size = 5

#Train set
data_dir = os.path.abspath("tableDataset")
dataset = datasets.ImageFolder(data_dir, transform=transform)

trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


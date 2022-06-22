import fiftyone as fo
import fiftyone.zoo as foz
from sklearn.metrics import label_ranking_average_precision_score
import torch
import torchvision
from torchvision import *
from torch import *
import matplotlib.pyplot as plt
import os 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# path del directorio de las imagenes
data_dir = os.path.abspath("tableDataset")


# Variable de transformación para el rescale y centering de las imagenes
transform = transforms.Compose(
    [transforms.Resize(255), transforms.CenterCrop(255), transforms.ToTensor()])

# Se carga las imagenes en la variable dataset

dataset = datasets.ImageFolder(data_dir, transform=transform)

# Loader de las imagenes

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1295,
                                         shuffle=True)

images, labels = next(iter(dataloader))

print('Numero de muestras : ', len(images))
image = images[2][0]

plt.imshow(image)

print("Image Size: ", image.size())
print("Dataset : ", dataset)
print(labels)

model = torchvision.models.detection.faster_rcnn(dataset)

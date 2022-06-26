
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 25, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

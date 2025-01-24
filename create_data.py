import json
import os

import pickle

from collections import OrderedDict

from tqdm import tqdm

from PIL import Image

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision.transforms import functional as Ft
from torchvision.models import get_model

import pygpc


## Image classigication model ##
class Model(nn.Module):

    def __init__(self, model_checkpoint, num_classes, dropout_prob):
        super(Model, self).__init__()
        self.model = model = get_model(model_checkpoint)

        # Modify the first convolutional layer to accept grayscale (1 channel) images
        # The original conv1 layer has in_channels=3, we change it to 1
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=model.conv1.bias,
        )
        # Replace the fully connected layer with a new one that includes dropout
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_prob), nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


model = Model(model_checkpoint="resnet18", num_classes=6, dropout_prob=0.5)
checkpoint = torch.load(
    "./data/weights/resnet18_epoch_3.pth",
    weights_only=False,
    map_location=torch.device("cpu"),
)
model.load_state_dict(checkpoint["model_state_dict"])

## Pygpc parameters ##
parameters = OrderedDict()
parameters["Rotation [deg]"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-90, 90])
parameters["Brightness"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 2])
parameters["Tilt [deg]"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-20, 20])

grid = pygpc.LHS(
    parameters_random=parameters, options={"criterion": "ese"}, n_grid=10000
)
coords = grid.coords

width = 244
height = 244
rotations = coords[:, 0]
brightnesses = coords[:, 1]
tilts = coords[:, 2]

## Sample model ##
class ImageTransformDataset:

    def __init__(self, image, width, height, rotations, brightnesses, tilts):
        self.image = image
        self.width = width
        self.height = height
        self.rotations = rotations
        self.brightnesses = brightnesses
        self.tilts = tilts
        self.length = min(
            (len(self.rotations), len(self.brightnesses), len(self.tilts))
        )

    def make_transform(self, rotation, brightness, tilt):

        def transform(image):
            return Ft.normalize(
                Ft.to_tensor(
                    Ft.resize(
                        Ft.perspective(
                            Ft.rotate(
                                Ft.adjust_brightness(
                                    image, brightness_factor=brightness
                                ),
                                angle=rotation,
                            ),
                            startpoints=[
                                [0, 0],
                                [self.width, 0],
                                [self.width, self.height],
                                [0, self.height],
                            ],
                            endpoints=[
                                [np.tan(np.deg2rad(tilt)) * self.width, 0],
                                [self.width - np.tan(np.deg2rad(tilt)) * self.width, 0],
                                [self.width, self.height],
                                [0, self.height],
                            ],
                        ),
                        size=(self.height, self.width),
                    ),
                ),
                mean=0.5,
                std=0.5,
            )

        return transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rotation = self.rotations[index]
        brightness = self.brightnesses[index]
        tilt = self.tilts[index]
        transform = self.make_transform(rotation, brightness, tilt)
        return transform(self.image)


image_index = 0
image_directory = "./data/al5083"
mode = "test"

with open(f"{image_directory}/{mode}/{mode}.json", "r") as f:
    dataset_info = json.load(f)
image_data = [(key, value) for key, value in dataset_info.items()]

image_path, label = image_data[0]
image = Image.open(f"{image_directory}/{mode}/{image_path}")

dataset = ImageTransformDataset(image, width, height, rotations, brightnesses, tilts)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

model.eval()
all_outputs = []
all_probas = []
with torch.no_grad():
    for X in tqdm(dataloader):
        output = model(X)
        all_outputs.append(output.cpu().numpy())
        probas = F.softmax(output, dim=1)
        all_probas.append(probas.cpu().numpy())

results = np.concatenate(all_probas)[:, [label]]

## Save data ##
os.makedirs("./data/gPC", exist_ok=True)
np.save("./data/gPC/results.npy", results)
np.save("./data/gPC/coords.npy", coords)
with open("./data/gPC/parameters.pkl", "wb") as file:
    pickle.dump(parameters, file)

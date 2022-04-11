import pandas as pd
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import os
# 3D library imports

from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class CustomDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.paths_df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.paths_df)

    def __getitem__(self, idx):
        root_path = self.paths_df.paths[idx]
        img_nr = random.randint(0,7) # Get a random image from the render folder 
        img_path = root_path+'/renders/'+str(img_nr)+'.png'
        model_path = root_path+'/models/model_normalized.obj' # The following string part of path is always teh same for the whole dataset

        # Load and transform image 
        print(img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = self.transform(image=image)
        image = transform["image"]

        # Load 3D object and prepare pointcloud data
        verts, faces, _ = load_obj(model_path)
        test_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])

        # Differentiably sample 5k points from the surface of each mesh and then compute the loss.
        sample_test = sample_points_from_meshes(test_mesh, 5000)

        return image, sample_test




data_transform = A.Compose(
    [
        A.Resize(224, 224, p=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ,max_pixel_value=255.0,
            p=1.0,),
        ToTensorV2(),
    ]
)

cwd = os.getcwd()
print(cwd)
csv_path = cwd +'/paths.csv'

train_dataset = CustomDataset(csv_path=csv_path, transform=data_transform)
train_dataloader = DataLoader(train_dataset, batch_size=128 ,shuffle=True, num_workers=6)

loop = tqdm(enumerate(train_dataloader),total = len(train_dataloader))

for batch_idx, (data, features) in loop:
    print(data.size(), features.size())
    



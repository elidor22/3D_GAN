
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from dataloader import CustomDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from model import CycleGenerator
import torch
from torch import optim
from pytorch3d.loss import chamfer_distance # Our loss function

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

batch_size = 2

train_dataset = CustomDataset(csv_path=csv_path, transform=data_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size ,shuffle=True, num_workers=6)

loop = tqdm(enumerate(train_dataloader),total = len(train_dataloader))

# Next returns index, image and pointcloud
# idx, nxt = next(iter(loop))
# print(nxt[0].size())
# print(nxt[1].size())
model = CycleGenerator()
# res = model(nxt[0], batch_size)
# print(res.size())

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = chamfer_distance
# Train loop here

epoch = 1
with tqdm(loop, unit="batch") as tepoch:
        epoch_loss = 0.0
        batch_nr = 1
        for batch_idx, (image, point_cloud) in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            image, point_cloud = image.to(device), point_cloud.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            res = model(image, batch_size)
            print(res.size(), point_cloud.size())
            loss, _ = chamfer_distance(res, point_cloud)

            loss.backward()
            optimizer.step()

            epoch_loss += res.shape[0]*loss.item()
            tepoch.set_postfix(loss=loss.item(), epoch_loss_status = epoch_loss/batch_nr)
            batch_nr += 1
            break



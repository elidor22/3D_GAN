
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
criterion_mse = nn.MSELoss()
# Train loop here

train_loop = tqdm(enumerate(train_dataloader),total = len(train_dataloader))
def train_step(epoch):
    with tqdm(train_loop, unit="batch") as tepoch:
            epoch_loss = 0.0
            batch_nr = 1
            for batch_idx, (image, point_cloud) in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                image, point_cloud = image.to(device), point_cloud.to(device)
                # print(point_cloud.size())

                # zero the parameter gradients
                optimizer.zero_grad()

                res = model(image, batch_size)
                # print(res[0].size(), point_cloud.size())
                

                # Calculate loss per elements as chamfer loss from Pytorch3d doesn't support batches
                loss_cmph = 0 
                for i in range(batch_size):
                    # res = model(image, batch_size)
                    loss_chamfer, _ = criterion(res[i], point_cloud[i])
                    
                    loss_cmph += loss_chamfer
                    # loss_mse = criterion_mse(res, point_cloud)
                    # loss_mse_accumulated += loss_mse

                

                loss_chamfer = loss_cmph/batch_size
                loss_mse = loss_mse = criterion_mse(res, point_cloud)

                # Sum mse loss with chamfer
                loss_ls = [loss_chamfer, loss_mse]
                loss = sum(loss_ls)
                # print(loss)

                loss.backward()
                optimizer.step()

                epoch_loss += res.shape[0]*loss.item()
                tepoch.set_postfix(loss=loss.item(), epoch_loss_status = epoch_loss/batch_nr)
                batch_nr += 1

                # Save model
                # torch.save(model.state_dict(), f'model_archive/model_epoch{epoch}.pth')
                # break
            torch.save(model.state_dict(), f'model_archive/model_epoch{epoch}.pth')



epochs = 1
for t in range(epochs):
    train_step(t)
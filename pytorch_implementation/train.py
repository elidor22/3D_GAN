
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from dataloader import CustomDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F



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
train_dataloader = DataLoader(train_dataset, batch_size=1 ,shuffle=True, num_workers=6)

loop = tqdm(enumerate(train_dataloader),total = len(train_dataloader))

# Next returns index, image and pointcloud
idx, nxt = next(iter(loop))
print(nxt[0].size())
print(nxt[1].size())

# Train loop here
# for batch_idx, (data, features) in loop:
#     tqdm.write(str(data.size()), file = None)
#     break

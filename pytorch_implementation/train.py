
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from dataloader import CustomDataset
from torch.utils.data import DataLoader



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
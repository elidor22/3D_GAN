import os, shutil
from os import walk
from tqdm import tqdm
# Get all folders in the given path
path = '/media/elidor/CC98A71E98A70654/Ubuntu/3D_Data/02691156'
# d = []

def get_dirFolders(path):
    d = []
    for (dirpath, dirnames, filenames) in walk(path):
        d.extend(dirnames)
        break
    return d
print(get_dirFolders(path))



# Check if the render dir exists, create it, render images
# remove directory with read-only files

def render_onDirectory(path = '', ls = []):
    final_ls = []
    for dir in ls:
        pth = path + '/'+str(dir)
        final_ls.append(pth)


    for dir in final_ls:
        pth1 = dir+'/renders'
        if  not os.path.isdir(pth1):
            print('Creating path...')
            os.mkdir(pth1)
            
        run_path = 'blender -b --python utils/renderer.py -- '+ str(pth1)+' '+str(dir)+'/models/model_normalized.obj' # Last element makes the render quiet on terminal
            # os.system(f'blender -b --python render.py -- {pth}/models/model_normalized.obj')
        os.system(run_path)


# path = '/media/elidor/CC98A71E98A70654/Ubuntu/3D_Data/test' 
path = '/media/elidor/CC98A71E98A70654/Ubuntu/3D_Data/buses_shapenet'
directories = get_dirFolders(path)
# render_onDirectory(path=path, ls = directories)





# Try multithreading using pytorch dataset
from torch.utils.data import Dataset, DataLoader

def render_onDirectory2(path = '', dir = ''):
    pth = path + '/'+str(dir)


    pth1 = pth+'/renders'
    if  not os.path.isdir(pth1):
        print('Creating path...')
        os.mkdir(pth1)
    print(pth1)
    run_path = 'blender -b --python utils/renderer.py -- '+ str(pth1)+' '+str(pth)+'/models/model_normalized.obj' # Last element makes the render quiet on terminal
        # os.system(f'blender -b --python render.py -- {pth}/models/model_normalized.obj')
    os.system(run_path)

class CustomDataset(Dataset):
    def __init__(self, path,ls_path):
        self.path = path
        self.ls_path = ls_path

    def __len__(self):
        return len(self.ls_path)

    def __getitem__(self, idx):
        render_onDirectory2(path=path, dir = self.ls_path[idx])
        return 0






path = '/media/elidor/CC98A71E98A70654/Ubuntu/3D_Data/tables'
directories = get_dirFolders(path)

train_dataset = CustomDataset(path=path, ls_path= directories)
train_dataloader = DataLoader(train_dataset, batch_size=64 ,shuffle=False, num_workers=8)
print(len(train_dataloader))
for render in tqdm(train_dataloader):
    pass

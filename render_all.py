import os, shutil
from os import walk
import tqdm
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
            
        run_path = 'blender -b --python renderer.py -- '+ str(pth1)+' '+str(dir)+'/models/model_normalized.obj' # Last element makes the render quiet on terminal
            # os.system(f'blender -b --python render.py -- {pth}/models/model_normalized.obj')
        os.system(run_path)


path = '/media/elidor/CC98A71E98A70654/Ubuntu/3D_Data/test' 
# path = '/media/elidor/CC98A71E98A70654/Ubuntu/3D_Data/02691156'
directories = get_dirFolders(path)
render_onDirectory(path=path, ls = directories)
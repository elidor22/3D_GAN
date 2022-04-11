import pandas as pd
import os, shutil
from os import walk


def paths_to_csv(path):

    def get_dirFolders(path):
        d = []
        for (dirpath, dirnames, filenames) in walk(path):
            d.extend(dirnames)
            break
        return d

    paths = get_dirFolders(path)

    # Add the dirname to existing path
    for i in range(len(paths)):
        paths[i] = path+'/'+paths[i]
    paths_dict = {'paths':paths} 

    df = pd.DataFrame(paths_dict)
    df.to_csv('paths.csv', index=False)

# path = '/media/elidor/CC98A71E98A70654/Ubuntu/3D_Data/02691156' 
path = '/media/elidor/CC98A71E98A70654/Ubuntu/3D_Data/test' 
paths_to_csv(path)
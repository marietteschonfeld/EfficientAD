import os
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.folder import is_image_file
import pandas as pd
from Dataloaders.AD_Dataset import *

from typing import Optional, Callable, List, Tuple, Dict, Any


class VisA(AD_Dataset):
    def __init__(self,root,category,train,pin_memory=False, crop=False, normalize=False, square_image=False):

        super(VisA, self).__init__(root,"visa",category,train,pin_memory, crop=crop, normalize=normalize, square_image=square_image)
        if self.__len__() == 0:
            raise FileNotFoundError("found 0 files in {}\n".format(self.subset_split))
        

    def _find_paths(self):
        image_paths, mask_paths, targets = [], [], []

        split_df = pd.read_csv(self.dataset_root+'/split_csv/1cls.csv')
        split_df = split_df[split_df["object"] == self.category]
        split_df = split_df.fillna(value="")
        if self.train:
            split_df = split_df[split_df['split'] == 'train']
        else:
            split_df = split_df[split_df['split'] == 'test']

        def map_target(lab):
            map = {'normal':0, 'anomaly':1}
            return map[lab]

        split_df['label'] = split_df['label'].apply(map_target)

        image_paths = list(split_df['image'])
        image_paths = [self.dataset_root+"/"+image_path for image_path in image_paths]
        mask_paths = list(split_df['mask'])
        mask_paths = [self.dataset_root+"/"+mask_path if mask_path != "" else "" for mask_path in mask_paths]
        targets = list(split_df['label'])

        return image_paths, mask_paths, targets

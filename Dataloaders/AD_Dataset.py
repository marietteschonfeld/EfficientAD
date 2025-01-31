import os
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import is_image_file
from math import ceil
import torch
import torchvision.transforms.v2 as transforms
from abc import ABC, abstractmethod
from torchvision.transforms.v2.functional import to_dtype, to_image
from torchvision.tv_tensors import Mask


def _convert_label(x):
    return 0 if x == 0 else 1

class AD_Dataset(VisionDataset, ABC):

    datasets = {'mvtec_ad':'mvtec_anomaly_detection', 
                'mvtec_loco':'mvtec_loco_anomaly_detection',
                'visa':'VisA_20220922'}
    
    categories = {"mvtec_ad": ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper'],
                  "visa": ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'],
                  "mvtec_loco": ['breakfast_box','juice_bottle','pushpins','screw_bag','splicing_connectors']
                }
    
    image_sizes = {"mvtec_ad": {'bottle':(1024, 1024),
                                'cable':(1024, 1024),
                                'capsule':(1024, 1024),
                                'carpet':(1024, 1024),
                                'grid':(1024, 1024),
                                'hazelnut':(1024, 1024),
                                'leather':(1024, 1024),
                                'metal_nut':(1024, 1024),
                                'pill':(1024, 1024),
                                'screw':(1024, 1024),
                                'tile':(1024, 1024),
                                'toothbrush':(1024, 1024),
                                'transistor':(1024, 1024),
                                'wood':(1024, 1024),
                                'zipper':(1024, 1024)},

                  "visa": {'candle':(1284,1168), 
                           'capsules': (1500,1000),
                           'cashew': (1274, 1176), 
                           'chewinggum': (1342,1118),
                           'fryum': (1500,1000),
                           'macaroni1': (1500, 1000),
                           'macaroni2': (1500, 1000),
                           'pcb1': (1404,1070),
                           'pcb2': (1404,1070),
                           'pcb3': (1562, 960),
                           'pcb4': (1358, 1104),
                           'pipe_fryum': (1300, 1154)},

                  "mvtec_loco": {'breakfast_box': (1600, 1280),
                                 'juice_bottle': (800, 1600),
                                 'pushpins': (1700, 1000),
                                 'screw_bag': (1600, 1100),
                                 'splicing_connectors': (1700, 850)}
    }

    normal_str = 'good'
    mask_str = 'ground_truth'
    train_str = 'train'
    test_str = 'test'

    def __init__(self,root,dataset,category,train,pin_memory, crop=False, normalize=False, square_image=False):
        self.dataset_name = dataset
        self.category = category

        self.image_size = self.image_sizes[self.dataset_name][self.category]

        self.crop = crop
        self.normalize = normalize
        self.square_image = square_image
        transform, mask_transform, target_transform = self.get_transforms()

        super(AD_Dataset, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train
        self.mask_transform = mask_transform
        self.pin_memory = pin_memory

        self.dataset_root = os.path.join(self.root, self.datasets[self.dataset_name])
        self.category = category.lower()
        self.subset_root = os.path.join(self.dataset_root, self.category)
        self.subset_split = os.path.join(self.subset_root, self.train_str if self.train else self.test_str)

        self.image_paths, self.mask_paths, self.targets = self._find_paths()
        self.targets = [self.target_transform(i) for i in self.targets]
        if self.dataset_name != "mvtec_loco":
            if self.train:
                self.masks = []
            else:
                self.masks = [self.mask_transform(self._load_image(path, mask=True)) for path in self.mask_paths]

        if not os.path.exists(self.subset_root):
            raise FileNotFoundError('subset {} is not found, please set download=True to download it.')

        if self.__len__() == 0:
            raise FileNotFoundError("found 0 files in {}\n".format(self.subset_split))

        if self.pin_memory:
            self.data = [self._load_image(path) for path in self.image_paths]


    def get_transforms(self):  
        # self.image_height = ceil((256/max(self.image_size))*min(self.image_size))
        # transform = [transforms.Resize(size=self.image_height, max_size=max(self.image_height+1, 256))]
        # mask_transform = [transforms.Resize(size=self.image_height, max_size=max(self.image_height+1, 256))]
        self.image_height = 256
        if not self.square_image:
            transform = [transforms.Resize(size=self.image_height)]
        else:
            transform = [transforms.Resize(size=(self.image_height, self.image_height))]

        mask_transform = [transforms.Resize(size=self.image_height)]

        if self.crop:
            short_size = min(self.image_size)
            ratio = self.image_height/short_size
    
            resized_image_size = (int(self.image_size[0]*ratio), int(self.image_size[1]*ratio))
            cropped_image_size = (resized_image_size[0]-32, resized_image_size[1]-32)
            cropper = transforms.CenterCrop(size=cropped_image_size)
            transform.append(cropper)
            mask_transform.append(cropper)

        if self.normalize:
            transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        transform = transforms.Compose(transform)
        mask_transform = transforms.Compose(mask_transform)
        
        target_transform = transforms.Lambda(_convert_label)
        return transform, mask_transform, target_transform


    def __getitem__(self, idx):
        if self.pin_memory:
            image = self.data[idx]
        else:
            image = self._load_image(self.image_paths[idx])
        target = self.targets[idx]

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


    def __len__(self):
        return len(self.targets)


    @abstractmethod
    def _find_paths(self):
        pass


    def _load_image(self, path, mask=False):
        if not mask:
            image = Image.open(path).convert("RGB")
            return to_dtype(to_image(image), torch.float32, scale=True)
        else:
            if path is None or path == "":
                image = Image.new('L', size=self.image_size)
            else:
                image = Image.open(path).convert("L")

            # scaler = {'mvtec_ad': 1/255, 'visa': 255, 'mvtec_loco': 1/255}[self.dataset_name]
            # return Mask(to_image(image).squeeze()*255, dtype=torch.uint8)
            image = to_image(image).squeeze()
            image = torch.where(image>0, 255, 0)
            return Mask(image/255, dtype=torch.float32)
        

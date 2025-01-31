import os
from PIL import Image
from torchvision.datasets.folder import is_image_file
from Dataloaders.AD_Dataset import *

class MVTecAD(AD_Dataset):

    def __init__(self,root,category,train,pin_memory,crop=False, normalize=False):
        super(MVTecAD, self).__init__(root,"mvtec_ad",category,train,pin_memory,crop=crop,normalize=normalize)

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.remove(self.normal_str)
        classes = [self.normal_str] + classes
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _find_paths(self):
        classes, class_to_idx = self._find_classes(self.subset_split)
        folder = self.subset_split

        image_paths, mask_paths, targets = [], [], []

        def find_mask_from_image(target_class, image_path):
            if target_class is self.normal_str:
                mask_path = None
            else:
                mask_path = image_path.replace(self.test_str, self.mask_str)
                fext = '.' + fname.split('.')[-1]
                mask_path = mask_path.replace(fext, '_mask' + fext)
            return mask_path

        for target_class in class_to_idx.keys():
            class_idx = class_to_idx[target_class]
            target_folder = os.path.join(folder, target_class)
            for root, _, fnames in sorted(os.walk(target_folder, followlinks=True)):
                for fname in fnames:
                    if is_image_file(fname):
                        image_paths.append(os.path.join(root, fname))
                        mask_paths.append(find_mask_from_image(target_class, image_paths[-1]))
                        targets.append(class_idx)

        return image_paths, mask_paths, targets

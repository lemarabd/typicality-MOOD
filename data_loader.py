from torch.utils.data import TensorDataset, DataLoader, Dataset
from os import listdir
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import math

class ISIC(Dataset):
    def __init__(self, cf, benign=True, test=False, gray=False, standardize=True, normalize=True, validation=False) -> None:
        super().__init__()
        self.cf = cf
        self.gray = gray
        self.standardize = standardize
        self.normalize = normalize

        benign_dir = cf.path + "/benign/"
        print('benign dir:', benign_dir)
        malignant_dir = cf.path + "/malignant/"
        print('malignant dir:', malignant_dir)

        if test:
            benign_dir = cf.path + "/test/benign/"
            malignant_dir = cf.path + "/test/malignant/"

        # load images and filenames
        benign_images = [(self.load_image(benign_dir + f), f) for f in listdir(benign_dir)]
        maligant_images = [(self.load_image(malignant_dir + f), f) for f in listdir(malignant_dir)]

        if benign:
            if validation:
                self.images = benign_images[:int(0.2 * len(benign_images))]
            else:
                self.images = benign_images[int(0.2 * len(benign_images)):]
            print("Number of healthy images: ", len(self.images))  
        else:
            # load malignant images
            if validation: 
                self.images = maligant_images[:int(0.2 * len(maligant_images))]
            else:
                self.images = maligant_images[int(0.2 * len(maligant_images)):]
            print("Number of unhealthy images: ", len(self.images))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image, filename = self.images[index]
        image = transforms.functional.resize(image, tuple(self.cf.patch_size)) 

        if self.cf.augmentation:
            if self.cf.affineTransform:
                angle, translate, scale, shear = transforms.RandomAffine.get_params(degrees=self.cf.degrees,
                                    translate=self.cf.translate, scale_ranges=self.cf.scale_ranges, shears=self.cf.shears, img_size=tuple(self.cf.patch_size))
    
                image = TF.affine(image, angle, translate, scale, shear)

        if not self.gray and self.normalize: # normalize per channel
            image = transforms.functional.to_tensor(image)
            mean = [0.6776, 0.4431, 0.4572]
            std = [0.3119, 0.2358, 0.2495]
            image = transforms.functional.normalize(image, mean, std)    
        elif not self.gray and not self.normalize:
            image = transforms.functional.to_tensor(image)
        else:
            image = transforms.functional.to_grayscale(image) # normalize grayscale image
            image = transforms.functional.to_tensor(image)
            image = transforms.functional.normalize(image, (0.4431,), (0.2358,))

        return image, filename
    
    def load_image(self, file_path):
        with Image.open(file_path) as img:
            return img.copy()

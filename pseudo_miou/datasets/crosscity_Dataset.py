# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import os
import torch
import imageio

from datasets.cityscapes_Dataset import City_Dataset, to_tuple
imageio.plugins.freeimage.download()

#crosscity_set_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
crosscity_set_13 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
synthia_set_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]

class CrossCity_Dataset(City_Dataset):
    def __init__(
        self,
        root='/local_datasets/NTHU_Datasets',
        list_path='./datasets/NTHU_list/',
        split='train',
        base_size=769,
        crop_size=769,
        training=True,
        class_13=True,
        class_16=False,
        random_mirror=False,
        random_crop=False,
        resize=False,
        gaussian_blur=False,
        city='Rio'
    ):

        # Args
        self.city=city
        self.set=split
        self.files = []
        self.data_path = root
        self.list_path = list_path
        self.split = split
        self.base_size = to_tuple(base_size)
        self.crop_size = to_tuple(crop_size)
        self.training = training
        self.class_16 = class_16
        self.class_13 = class_13

        # Augmentations
        self.random_mirror = random_mirror
        self.random_crop = random_crop
        self.resize = resize
        self.gaussian_blur = gaussian_blur

        # Files
        item_list_filepath = os.path.join(self.list_path,city,"List", self.split + ".txt")
        print("item_list_filepath : ",item_list_filepath)
        if not os.path.exists(item_list_filepath):
            raise Warning("split must be train/val/trainavl/test")
     #   self.image_filepath = os.path.join(self.data_path,city,'Images',self.split.capitalize())
     #   self.gt_filepath = os.path.join(self.data_path,city,'Labels',self.split.capitalize(),name[:-4] + "_eval.png")
        self.items = [id.strip() for id in open(item_list_filepath)]

        # Label map
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 19: 3, 20: 4, 21: 5, 23: 6,
                              24: 7, 25: 8, 26: 9, 28: 10, 32: 11, 33: 12}
    
        # Only consider 13 shared classes
        self.class_13 = class_13
        self.trainid_to_13id = {id: i for i, id in enumerate(crosscity_set_13)}
        
        for name in self.items:
            if self.set=='train':
                img_file = os.path.join(self.data_path,city,'Images',self.split.capitalize(), name+'.jpg')
            else:
                img_file = os.path.join(self.data_path,city,'Images',self.split.capitalize(), name)
            if self.set == "train":
                self.files.append({
                    "img": img_file,
                    "label": "",
                    "name": name
                })
            else:
                label_file = os.path.join(self.data_path,city,'Labels',self.split.capitalize(), name[:-4] + "_eval.png")
                self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                }) 

        print("{} num images in Crosscity {} {} set have been loaded.".format(
            len(self.items), self.city, self.split))

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        image1 = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]
        """
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        

        image = np.asarray(image, np.float32)

        image = image[:, :, ::-1]  # change to BGR
        image = image / 255.0
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        """
        if self.set == "train":
            label = datafiles["label"]
            label_copy = label
            image = self._train_sync_transform_crosscity(image)
        else:
            
            label = Image.open(datafiles["label"])
            """
            label = label.resize(self.crop_size, Image.NEAREST)
            label = np.asarray(label, np.float32)
            label_copy = (-1) * np.ones(label.shape, dtype=np.float32)
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
            target=label_copy.copy()
            label_copy = torch.from_numpy(target)
            image=self._img_transform(image)
            """
            image, label_copy = self._val_sync_transform(image1, label)

        return image, label_copy, name

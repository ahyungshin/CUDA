import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from PIL import Image
import numpy as np
import os
import torch
import imageio

from datasets.cityscapes_Dataset import City_Dataset, to_tuple
imageio.plugins.freeimage.download()
class IDDDataSet(City_Dataset):
    def __init__(
        self,
        root='/local_datasets/IDD',
        list_path='./datasets/idd_list/',
        split='train',
        base_size=769,
        crop_size=769,
        training=True,
        class_16=False,
        class_13= False,
        random_mirror=False,
        random_crop=False,
        resize=False,
        gaussian_blur=False,
    ):
        # Args
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
        item_list_filepath = os.path.join(self.list_path, self.split + ".txt")
        if not os.path.exists(item_list_filepath):
            raise Warning("split must be train/val/trainavl/test")
        self.items = [id.strip() for id in open(item_list_filepath)]
        #19classes
        #self.id_to_trainid = {0: 0, 3: 1, 29: 2, 20: 3, 21: 4, 26: 5, 25: 6,
        #                          24: 7, 32: 8, 250:9,33: 10, 6: 11, 8: 12, 12: 13,
        #                          13: 14, 14: 15, 17: 16, 9: 17, 10: 18}
        print("{} num images in IDD {} set have been loaded.".format(
            len(self.items), self.split))
        self.files = []
        self.set = set
        """
        if self.class_16:
            print('num_class in loader : ',self.num_classes)
            self.id_to_trainid = {0: 0, 3: 1, 29: 2, 20: 3, 21: 4, 26: 5, 25: 6,
                                  24: 7, 32: 8, 33: 9, 6: 10, 8: 11, 12: 12,
                                  13: 13, 14: 14, 17: 15, 9: 16, 10: 17}
        else:
            raise NotImplementedError("Unavailable number of classes")
        """
        #18classes
        self.id_to_trainid = {0: 0, 3: 1, 29: 2, 20: 3, 21: 4, 26: 5, 25: 6,
                                  24: 7, 32: 8, 33: 9, 6: 10, 8: 11, 12: 12,
                                  13: 13, 14: 14, 17: 15, 9: 16, 10: 17}
        for name in self.items:
            #print(self.root, "leftImg8bit/%s/%s" % (self.split, name))
            #img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.split, name))
            img_file = osp.join(self.data_path, "leftImg8bit",self.split,name)
            #label_file = osp.join(self.root, "gtFine/%s/%s_gtFine_labelids.png" % (self.set, name[:-16]))
            label_file = osp.join(self.data_path, "gtFine",self.split,name[:-16]+"_gtFine_labelids.png")
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        gt_image = Image.open(datafiles["label"])
        name = datafiles["name"]

        """
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        image = image[:, :, ::-1]  # change to BGR
        image = image / 255.0
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        """
        if (self.split == "train") and self.training:
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)
        return image, gt_image, name
     #   return image, label_copy, name


if __name__ == '__main__':
    dst = IDDDataSet('working_directory/data/IDD', './idd_list/train.txt',
                     crop_size=(512, 256), ignore_label=255, set='train', num_classes=18)
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels, name = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()

import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image, ImageFile
import json
import cv2
import imageio
from datasets.cityscapes_Dataset import City_Dataset, to_tuple
imageio.plugins.freeimage.download()
ImageFile.LOAD_TRUNCATED_IMAGES = True

# CLASSES = ('Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier',
# 			'Wall', 'Bike Lane', 'Crosswalk - Plain', 'Curb Cut', 'Parking', 'Pedestrian Area',
# 			'Rail Track', 'Road', 'Service Lane', 'Sidewalk', 'Bridge', 'Building', 'Tunnel',
# 			'Person', 'Bicyclist', 'Motorcyclist', 'Other Rider', 'Lane Marking - Crosswalk',
# 			'Lane Marking - General', 'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation',
# 			'Water', 'Banner', 'Bench', 'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera',
# 			'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Pothole',
# 			'Street Light', 'Pole', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Light',
# 			'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can', 'Bicycle', 'Boat',
# 			'Bus', 'Car', 'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle', 'Trailer',
# 			'Truck', 'Wheeled Slow', 'Car Mount', 'Ego Vehicle', 'Unlabeled')

def _get_mapillary_pairs(folder,split='train'):
    def get_path_pairs(img_folder,mask_folder):
        img_paths = []
        mask_paths = []
        for root,_,files in os.walk(img_folder):
            for filename in files:
                if filename.endswith('.jpg'):
                    imgpath = os.path.join(root,filename)
                    # foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('.jpg','.png')
                    maskpath = os.path.join(mask_folder,maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or img:',imgpath,maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths),img_folder))
        return img_paths,mask_paths
    if split in ('train','val'):
        if split == 'train':
            split_root = 'training'
        else:
            split_root = 'validation'
        img_folder = os.path.join(folder,split_root+'/images')
        mask_folder = os.path.join(folder,split_root+'/v1.2/labels')
        img_paths,mask_paths = get_path_pairs(img_folder,mask_folder)
        return img_paths,mask_paths
    else:
        assert split == 'trainval'
        print('traintest set')
        train_img_folder = os.path.join(folder,'training/images')
        train_mask_folder = os.path.join(folder,'training/v1.2/labels')
        val_img_folder = os.path.join(folder,'validation/images')
        val_mask_folder = os.path.join(folder,'validation/v1.2/labels')
        train_img_paths,train_mask_paths = get_path_pairs(train_img_folder,train_mask_folder)
        val_img_paths,val_mask_paths = get_path_pairs(val_img_folder,val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths,mask_paths
class MapillaryDataSet(City_Dataset):
	def __init__(
        self,
        root='/local_datasets/MapillaryVistas',
        list_path='./datasets/vistas_list/',
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
		self.set=split
		self.files = []
		self.root = root
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
		"""
		item_list_filepath = os.path.join(self.list_path, self.split + ".txt")
		self.items = [id.strip() for id in open(item_list_filepath)]
		self.set = set
		if split=='train':
			self.split='training'
		elif split=='val':
			self.split='validation'
		"""
		self.images,self.mask_paths = _get_mapillary_pairs(self.root,self.split)
		assert(len(self.images) == len(self.mask_paths))
		if len(self.images) == 0:
			raise RuntimeError('Found 0 images in subfolders of : \
            ' + self.root + '\n')
		"""
		for img_name in self.img_ids:
			img_file = osp.join(self.root, "%s/images/%s" % (self.split, img_name))
			img_name = img_name[:-3]+'png' #.replace(".jpg", ".png") 
			label_file = osp.join(self.root, "%s/v1.2/labels/%s" % (self.split, img_name))
			self.files.append({
				"img": img_file,
				"label": label_file,
				"name": img_name
			})
		"""
		self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
		"""
		self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
							  3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
							  7: 0, 8: 0, 9: ignore_label, 10: ignore_label, 11: 1, 12: 1, 13: 1,
							  14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 2,
							  18: ignore_label, 19: 2, 20: 2, 21: 3, 22: 3, 23: 4, 24: 4, 25: 5, 26: 6, 27: 6,
							  28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 6}
		"""
		#19classes
		#self.id_to_trainid = {
        #    13: 0,24:0,41:0, 2: 1, 15: 1, 17: 2, 6: 3, 3: 4, 45: 5, 47: 5, 48: 6,50:7,30:8,29:9,
        #    27: 10, 19: 11, 20: 12,21:12,22:12, 55: 13, 61: 14, 54: 15, 58: 16, 57: 17, 52: 18
        #}	
		#18classes
		self.id_to_trainid = {
            13: 0,24:0,41:0, 2: 1, 15: 1, 17: 2, 6: 3, 3: 4, 45: 5, 47: 5, 48: 6,50:7,30:8,
            27: 9, 19: 10, 20: 11,21:11,22:11, 55: 12, 61: 13, 54: 14, 58: 15, 57: 16, 52: 17
        }		
				  
	def __len__(self):
		return len(self.images)


	def __getitem__(self, index):
		image = Image.open(self.images[index]).convert('RGB')
		gt_image = Image.open(self.mask_paths[index])
		name='AA'
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
	dst = MapillaryDataSet("./data", is_transform=True)
	trainloader = data.DataLoader(dst, batch_size=4)
	for i, data in enumerate(trainloader):
		imgs, labels = data
		if i == 0:
			img = torchvision.utils.make_grid(imgs).numpy()
			img = np.transpose(img, (1, 2, 0))
			img = img[:, :, ::-1]
			plt.imshow(img)
			plt.show()

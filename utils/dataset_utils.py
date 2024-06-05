import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
import numpy as np
from PIL import ImageEnhance
import cv2

import io

def CustomResizeCrop(img, gt, size):
    for i in range(10):  
        scale_factor = random.uniform(0.5, 2.0)
        h, w = img.size
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        if new_h < size or new_w < size:
            continue  
            
        img_resized = img.resize((new_w, new_h), Image.BICUBIC)
        gt_resized = gt.resize((new_w, new_h), Image.NEAREST)

        for j in range(10):  
            x = random.randint(0, new_w - size)
            y = random.randint(0, new_h - size)
            
            crop_image = img_resized.crop((x, y, x + size, y + size))
            crop_gt = gt_resized.crop((x, y, x + size, y + size))
            
            gt_array = np.array(crop_gt)
            white_ratio = np.sum(gt_array == 255) / (size * size)
            
            if white_ratio > 0.05 and white_ratio < 0.75:
                return crop_image, crop_gt

    return img, gt

def RandomFlip(img, gt):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
    return img, gt


def jpeg_compression(img, min_quality, max_quality):
    
    quality = random.randint(min_quality, max_quality)
    
    image_bytes = io.BytesIO()
    
    img.save(image_bytes, format='JPEG', quality=quality)
    
    compressed_image_data = image_bytes.getvalue()
    
    compressed_image = Image.open(io.BytesIO(compressed_image_data))
    
    return compressed_image
  
class IFLDataset(data.Dataset):
    def __init__(self, data_dir, img_size, mode):
        self.img_size = img_size
        self.mode = mode
        image_root = os.path.join(data_dir, mode, 'img')
        print(image_root)
        gt_root = os.path.join(data_dir, mode, 'gt')
        self.images = [image_root + '/' + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.TIF') or f.endswith('.png')]
        self.gts = [gt_root + '/' + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.tif')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        if self.mode == 'train':
            image, gt = RandomFlip(image, gt)
            image, gt = CustomResizeCrop(image, gt, self.img_size)
            if random.random() < 0.5:
                image = jpeg_compression(image, 70, 95)

            image = self.img_transform(image)
            gt = self.gt_transform(gt)
        else:
            image = self.img_transform(image)
            gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f) 
            img_array = np.array(img)
            threshold = 128
            binary_img_array = np.where(img_array > threshold, 255, 0)
            img = Image.fromarray(np.uint8(binary_img_array))
            return img.convert('L')

    def __len__(self):
        return self.size


def get_train_loader(data_dir, mode, batch_size, num_workers, img_size, shuffle=True):
    
    dataset = IFLDataset(data_dir, img_size, mode)
    tain_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)
    return tain_loader

def get_val_loader(data_dir, mode, batch_size, num_workers, img_size, shuffle=False):

    dataset = IFLDataset(data_dir, img_size, mode)
    val_loader = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers)
    return val_loader


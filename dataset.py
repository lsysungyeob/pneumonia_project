import itk
import os
import torch
import numpy as np
import itk
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# ================================
# 1. 데이터셋 정의
# ================================
class ITKDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(256, 256), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self.mask_paths = sorted(os.listdir(mask_dir))
        self.target_size = target_size
        self.transform = transform

    def _resize_image(self, image_np, size):
        """
        이미지 크기 조정 (torchvision.transforms.functional 사용)
        """
        image_tensor = torch.tensor(image_np, dtype=torch.float32)
        resized_image = TF.resize(image_tensor, size)
        return resized_image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])

        itk_image = itk.imread(image_path, itk.UC)
        itk_mask = itk.imread(mask_path, itk.UC)
        
        #origin_size = itk.size(itk_image)

        image_np = itk.GetArrayViewFromImage(itk_image).astype(np.float32) / 255.0
        mask_np = itk.GetArrayViewFromImage(itk_mask).astype(np.float32) / 255.0

        # 크기 조정
        image_np = np.expand_dims(image_np, axis=0)
        mask_np = np.expand_dims(mask_np, axis=0)
        image_resized = self._resize_image(image_np, self.target_size)
        mask_resized = self._resize_image(mask_np, self.target_size)

        if self.transform:
            image_resized = self.transform(image_resized)
            mask_resized = self.transform(mask_resized)

        return image_resized, mask_resized

class ITKTestset(Dataset):
    def __init__(self, image_dir, target_size=(256, 256), transform=None):
        self.image_dir = image_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self.target_size = target_size
        self.transform = transform

    def _resize_image(self, image_np, size):
        """
        이미지 크기 조정 (torchvision.transforms.functional 사용)
        """
        image_tensor = torch.tensor(image_np, dtype=torch.float32)
        resized_image = TF.resize(image_tensor, size)
        return resized_image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_paths[idx])

        itk_image = itk.imread(image_path, itk.UC)
        
        origin_size = itk.size(itk_image)
        w, h = tuple(origin_size)

        image_np = itk.GetArrayViewFromImage(itk_image).astype(np.float32) / 255.0

        # 크기 조정
        image_np = np.expand_dims(image_np, axis=0)
        image_resized = self._resize_image(image_np, self.target_size)

        if self.transform:
            image_resized = self.transform(image_resized)


        return image_resized, image_np, (h, w)
    
class RESNETDataset(Dataset):
    def __init__(self, model, device, normal_dir, pneumonia_dir, target_size=(256, 256), transform=None):
        self.model = model
        self.device = device
        self.normal_dir = normal_dir
        self.pneumonia_dir = pneumonia_dir
        #self.image_paths = sorted(os.listdir(image_dir))
        self.target_size = target_size
        self.transform = transform
        
        self.images = []
        self.labels = []
        
        for image_name in os.listdir(self.normal_dir):
            self.images.append(os.path.join(self.normal_dir, image_name))
            self.labels.append(0)
        
        for image_name in os.listdir(self.pneumonia_dir):
            self.images.append(os.path.join(self.pneumonia_dir, image_name))
            self.labels.append(1)

    def _resize_image(self, image_np, size):
        """
        이미지 크기 조정 (torchvision.transforms.functional 사용)
        """
        image_tensor = torch.tensor(image_np, dtype=torch.float32)
        resized_image = TF.resize(image_tensor, size)
        return resized_image
    
    def apply_mask_to_image(self, image, mask, check):
        # 마스크 이진화: 127 이상이면 1, 미만이면 0
        binary_mask = (mask > check).astype(np.uint8)

        # 마스크 적용 (마스크가 1인 부분만 남김)
        masked_image = image * binary_mask  # element-wise 곱셈

        return masked_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        itk_image = itk.imread(image_path, itk.UC)
        
        origin_size = itk.size(itk_image)
        w, h = tuple(origin_size)

        image_np = itk.GetArrayViewFromImage(itk_image).astype(np.float32) / 255.0

        # 크기 조정
        image_origin = np.expand_dims(image_np, axis=0)
        image_resized = self._resize_image(image_origin, self.target_size)

        if self.transform:
            image_resized = self.transform(image_resized)
            
        image_resized = image_resized.to(self.device)
        image_resized = image_resized.unsqueeze(0)
        
        #print(image_resized)
            
        output = self.model(image_resized)
        mask = torch.sigmoid(output).cpu()
        
        mask_resize = TF.resize(mask, (h, w), antialias=True)
        mask_resize = (mask_resize.squeeze().detach().numpy() * 255).astype(np.uint8)
        #image_np = (image_np.cpu().squeeze().numpy() * 255).astype(np.uint8)
        image_np = (image_origin * 255).astype(np.uint8)
        
        masked_image = self.apply_mask_to_image(image_np, mask_resize, 50)
        
        #masked_image = np.expand_dims(masked_image, axis=0)
        masked_resize = self._resize_image(masked_image, self.target_size)

        return masked_resize, label
    
class RESNETTestset(Dataset):
    def __init__(self, model, device, test_dir, target_size=(256, 256), transform=None):
        self.model = model
        self.device = device
        self.test_dir = test_dir
        self.target_size = target_size
        self.transform = transform
        
        self.images = []
        
        for image_name in os.listdir(self.test_dir):
            self.images.append(os.path.join(self.test_dir, image_name))

    def _resize_image(self, image_np, size):
        """
        이미지 크기 조정 (torchvision.transforms.functional 사용)
        """
        image_tensor = torch.tensor(image_np, dtype=torch.float32)
        resized_image = TF.resize(image_tensor, size)
        return resized_image
        
    def apply_mask_to_image(self, image, mask, check):
        # 마스크 이진화: 127 이상이면 1, 미만이면 0
        binary_mask = (mask > check).astype(np.uint8)

        # 마스크 적용 (마스크가 1인 부분만 남김)
        masked_image = image * binary_mask  # element-wise 곱셈

        return masked_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        itk_image = itk.imread(image_path, itk.UC)
        
        origin_size = itk.size(itk_image)
        w, h = tuple(origin_size)

        image_np = itk.GetArrayViewFromImage(itk_image).astype(np.float32) / 255.0

        # 크기 조정
        image_origin = np.expand_dims(image_np, axis=0)
        image_resized = self._resize_image(image_origin, self.target_size)

        if self.transform:
            image_resized = self.transform(image_resized)
            
        image_resized = image_resized.to(self.device)
        image_resized = image_resized.unsqueeze(0)
            
        output = self.model(image_resized)
        mask = torch.sigmoid(output).cpu()
        
        mask_resize = TF.resize(mask, (h, w), antialias=True)
        mask_resize = (mask_resize.squeeze().detach().numpy() * 255).astype(np.uint8)
        image_np = (image_origin * 255).astype(np.uint8)
        
        masked_image = self.apply_mask_to_image(image_np, mask_resize, 50)
        
        masked_resize = self._resize_image(masked_image, self.target_size)

        return masked_resize
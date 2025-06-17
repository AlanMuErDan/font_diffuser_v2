import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def get_nonorm_transform(resolution):
    nonorm_transform =  transforms.Compose(
            [transforms.Resize((resolution, resolution), 
                               interpolation=transforms.InterpolationMode.BILINEAR), 
             transforms.ToTensor()])
    return nonorm_transform

def augment_image(img: Image.Image) -> Image.Image:
    """
    对单张 96×96 灰度图做随机缩放和平移，填充白底。
    """
    # 1. 随机缩放比例
    scale = random.uniform(0.5, 1.0)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)

    # 2. 缩放
    scaled = img.resize((new_w, new_h), resample=Image.BILINEAR)

    # 3. 计算中心点可选范围
    half_w_floor = new_w // 2
    half_h_floor = new_h // 2
    half_w_ceil = new_w - half_w_floor
    half_h_ceil = new_h - half_h_floor

    cx_min = half_w_floor
    cx_max = img.width - half_w_ceil
    cy_min = half_h_floor
    cy_max = img.height - half_h_ceil

    # 4. 随机中心点
    center_x = random.randint(cx_min, cx_max)
    center_y = random.randint(cy_min, cy_max)

    # 5. 由中心点转为左上角坐标
    left = center_x - half_w_floor
    top  = center_y - half_h_floor

    # 6. 新建白底画布，并把缩放图贴上去
    background = Image.new('L', (img.width, img.height), color=255)
    background.paste(scaled, (left, top))

    return background


class FontDataset(Dataset):
    """The dataset of font generation  
    """
    def __init__(self, args, phase, transforms=None, scr=False):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.scr = scr
        if self.scr:
            self.num_neg = args.num_neg
        self.augment_content = getattr(args, "augment_content", False)
        self.augment_style = getattr(args, "augment_style", False)
        self.augment_content_prob = getattr(args, "augment_content_prob", 0.5)
        self.augment_style_prob = getattr(args, "augment_style_prob", 0.5)
        
        # Get Data path
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

    def get_path(self):
            self.target_images = []
            self.style_to_images = {}
            target_image_dir = f"{self.root}/{self.phase}/TargetImage"
            
            for style in os.listdir(target_image_dir):
                style_path = os.path.join(target_image_dir, style)
                if not os.path.isdir(style_path):
                    continue  # Skip files like .DS_Store
                    
                images_related_style = []
                for img in os.listdir(style_path):
                    img_path = os.path.join(style_path, img)
                    self.target_images.append(img_path)
                    images_related_style.append(img_path)
                
                self.style_to_images[style] = images_related_style

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        target_image_name = target_image_path.split('/')[-1]
        style, content = target_image_name.split('.')[0].split('+')
        
        # Read content image
        content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.jpg"
        content_image = Image.open(content_image_path).convert('RGB')

        # Augment content image
        if self.augment_content and random.random() < self.augment_content_prob:
            content_image = augment_image(content_image).convert("RGB")

        # Random sample used for style image
        images_related_style = self.style_to_images[style].copy()
        images_related_style.remove(target_image_path)
        style_image_path = random.choice(images_related_style)
        style_image = Image.open(style_image_path).convert("RGB")

        # Augment style image
        if self.augment_style and random.random() < self.augment_style_prob:
            style_image = augment_image(style_image).convert("RGB")
        
        # Read target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)
        
        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image}
        
        if self.scr:
            # Get neg image from the different style of the same content
            style_list = list(self.style_to_images.keys())
            style_index = style_list.index(style)
            style_list.pop(style_index)
            choose_neg_names = []
            for i in range(self.num_neg):
                choose_style = random.choice(style_list)
                choose_index = style_list.index(choose_style)
                style_list.pop(choose_index)
                choose_neg_name = f"{self.root}/train/TargetImage/{choose_style}/{choose_style}+{content}.jpg"
                choose_neg_names.append(choose_neg_name)

            # Load neg_images
            for i, neg_name in enumerate(choose_neg_names):
                neg_image = Image.open(neg_name).convert("RGB")
                if self.transforms is not None:
                    neg_image = self.transforms[2](neg_image)
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
            sample["neg_images"] = neg_images

        return sample

    def __len__(self):
        return len(self.target_images)

# Import Python Standard Library dependencies
import datetime
from functools import partial
from glob import glob
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
from typing import Any, Dict, Optional



# Import utility functions
from cjm_psl_utils.core import download_file, file_extract, get_source_code
from cjm_pil_utils.core import resize_img, get_img_files, stack_imgs
from cjm_pytorch_utils.core import pil_to_tensor, tensor_to_pil, get_torch_device, set_seed, denorm_img_tensor, move_data_to_device
from cjm_pandas_utils.core import markdown_to_pandas, convert_to_numeric, convert_to_string
from cjm_torchvision_tfms.core import ResizeMax, PadSquare, CustomRandomIoUCrop

# Import the distinctipy module
from distinctipy import distinctipy

# Import matplotlib for creating plots
import matplotlib.pyplot as plt

# Import numpy
import numpy as np

# Import the pandas package
import pandas as pd
import cv2
# Set options for Pandas DataFrame display
pd.set_option('max_colwidth', None)  # Do not truncate the contents of cells in the DataFrame
pd.set_option('display.max_rows', None)  # Display all rows in the DataFrame
pd.set_option('display.max_columns', None)  # Display all columns in the DataFrame

# Import PIL for image manipulation
from PIL import Image, ImageDraw
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from engine import train_one_epoch, evaluate

# Import PyTorch dependencies
import torch
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtnt.utils import get_module_summary
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.transforms.v2  as transforms
from torchvision.transforms.v2 import functional as TF
from torchvision import tv_tensors
# Import Mask R-CNN
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.ops
from torchvision.ops import masks_to_boxes
import utils
import transforms as T
import albumentations as A
import cv2
import time
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.io import read_image 
# Import tqdm for progress bar
from tqdm.auto import tqdm

device = get_torch_device()
dtype = torch.float32


# Display the first five entries from the dictionary using a Pandas DataFrame






def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)
NORMALIZE = False 
def getMasks(annotBox, shape, color=1):

    img = np.zeros(shape,dtype=np.float32)

    for annotations in annotBox:
        low = annotations[0],annotations[1]
        high= annotations[2],annotations[3]
        img[low:high] = color
    return img
class Normalize:
    def __call__(self, image, target):
        image = TF.normalize(image, RESNET_MEAN, RESNET_STD)
        return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
class ToTensor:
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        return image, target
def get_transform(train):
    transforms = [ToTensor()]
    if NORMALIZE:
        transforms.append(Normalize())
    

    return Compose(transforms)
class CraterDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, "images"))))
        self.annots = list(sorted(os.listdir(os.path.join(self.root, "labels"))))
        self.classes = ['Background','Crater']
        
        
    # Converts boundry box formats, this version assumes single class only!
    def convert_box_cord(self,bboxs, format_from, format_to, img_shape):
        if format_from == 'normxywh':
            if format_to == 'xyminmax':
                xw = bboxs[:, (1, 3)] * img_shape[1]
                yh = bboxs[:, (2, 4)] * img_shape[0]
                xmin = xw[:, 0] - xw[:, 1] / 2
                xmax = xw[:, 0] + xw[:, 1] / 2
                ymin = yh[:, 0] - yh[:, 1] / 2
                ymax = yh[:, 0] + yh[:, 1] / 2

                coords_converted = np.column_stack((xmin, ymin, xmax, ymax))
        return coords_converted
    
    def parse_labels_from_file(file_path):
     labels = []
     with open(file_path, 'r') as file:
        for line in file:
            label_parts = line.strip().split()
            if len(label_parts) == 5:
                class_id, x, y, w, h = map(float, label_parts)
                labels.append((int(class_id), x, y, w, h))
     return labels
    def create_polygon_mask(self,image_size, vertices):
        mask_img = Image.new('L', image_size, 0)
        pt1 = vertices[0],vertices[1]
        pt2 = vertices[2],vertices[3]
        pt3 = vertices [2],vertices[1]
        pt4 = vertices [0],vertices[3]
        verticeTuples = [pt1,pt2,pt3,pt4]
    # Draw the polygon on the image. The area inside the polygon will be white (255).
        ImageDraw.Draw(mask_img, 'L').polygon(verticeTuples, fill=(255))

    # Return the image with the drawn polygon
        return mask_img

    


    def __getitem__(self, idx):
        # load images and boxes
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annot_path = os.path.join(self.root, "labels", self.annots[idx])
        img = cv2.imread(img_path)
        img = Image.open(img_path).convert("RGB")
        
        

      



        # retrieve bbox list and format to required type,
        # if annotation file is empty, fill dummy box with label 0
        if (os.path.getsize(annot_path) != 0):
            bboxs = np.loadtxt(annot_path, ndmin=2)
            bboxs = self.convert_box_cord(bboxs, 'normxywh', 'xyminmax', (img.height,img.width))
            num_objs = len(bboxs)
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            # suppose all instances are not crowd
    

            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            maskImgs = [self.create_polygon_mask((img.height,img.width),xy)for xy in bboxs]
            masks = Mask(torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in maskImgs]))
            bboxs = torch.as_tensor(bboxs, dtype=torch.float32)
            image_id = torch.tensor([idx])
            
            area = (bboxs[:, 3] - bboxs[:, 1]) * (bboxs[:, 2] - bboxs[:, 0])
            image_id = torch.tensor([idx])
            target = {}
            target["boxes"] = bboxs
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = idx
            target["area"] = area
            target["iscrowd"] = iscrowd
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img,target
            # target["iscrowd"] = iscrowd
        else:
            bboxs = torch.as_tensor([[0, 0, 416, 416]], dtype=torch.float32)
            area = (bboxs[:, 3] - bboxs[:, 1]) * (bboxs[:, 2] - bboxs[:, 0])
            image_id = torch.tensor([idx])
            maskImgs = [self.create_polygon_mask((img.height,img.width),xy)for xy in bboxs]
            masks = Mask(torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in maskImgs]))
            labels = torch.zeros((1,), dtype=torch.int64)
            iscrowd = torch.zeros((1,), dtype=torch.int64)
            target = {}
            target["boxes"] = bboxs
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = idx
            target["area"] = area
            target["iscrowd"] = iscrowd
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img,target

        


        
       

    def __len__(self):
        return len(self.imgs)




   
import utils

dataset = CraterDataset('LU3M6TGT_yolo_format/train',get_transform(train=True))
dataset_val = CraterDataset('LU3M6TGT_yolo_format/valid',get_transform(train=False))


# Set the number of worker processes for loading data.

# Define parameters for DataLoader
train_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=4,
    collate_fn=utils.collate_fn
)
data_loader_test = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=utils.collate_fn
)
# For Training




model = get_model_instance_segmentation(2)
model.to(device)



params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=20,
    gamma=0.2
)

num_epochs = 20

result_mAP = []
best_epoch = None
mTrain = False
if(mTrain):

    for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
    # update the learning rate
        lr_scheduler.step()
    # evaluate on the test dataset
        results = evaluate(model, data_loader_test, device=device)
        result_mAP.append(results.coco_eval['bbox'].stats[1])
        if result_mAP[-1] == max(result_mAP):
            best_save_path = os.path.join(f'Crater_bestmodel_noaug_sgd(wd=0)_8batch-epoch{epoch}.pth')
            torch.save(model.state_dict(), best_save_path)
            best_epoch = int(epoch)
            print(f'\n\nmodel from epoch number {epoch} saved!\n result is {max(result_mAP)}\n\n')

    save_path = os.path.join(f'Crater_noaug_sgd_2batch-lastepoch{num_epochs-1}.pth')
    torch.save(model.state_dict(), save_path)
    print(f'model from last epoch(no.{num_epochs-1}) saved')
else:
    dataset_test = CraterDataset('LU3M6TGT_yolo_format/valid', get_transform(train=False))

    data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=2,
    collate_fn=utils.collate_fn)
    savedModel = get_model_instance_segmentation(2)
    savedModel.load_state_dict(torch.load(os.path.join(f'Crater_bestmodel_noaug_sgd(wd=0)_8batch-epoch{19}.pth'),map_location=device))
    savedModel.to(device)
    evaluate(savedModel, data_loader_test, device=device)
    color_inference = np.array([0.0,0.0,255.0])
    color_label = np.array([255.0,0.0,0.0])

# Score value thershold for displaying predictions
    detection_threshold = 0.7
# to count the total number of images iterated through
    frame_count = 0
# to keep adding the FPS for each image
    total_fps = 0
for i,data in enumerate(data_loader_test):
    # get the image file name for predictions file name
    image_name = 'image no:' + str(int(data[1][0]['image_id']))
    model_image = data[0][0]
    cv2_image = np.transpose(model_image.numpy()*255,(1, 2, 0)).astype(np.float32)
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR).astype(np.float32)

    # add batch dimension
    model_image = torch.unsqueeze(model_image, 0)
    start_time = time.time()
    with torch.no_grad():
        outputs = savedModel(model_image.to(device))
    end_time = time.time()
    # get the current fps
    fps = 1 / (end_time - start_time)
    # add `fps` to `total_fps`
    total_fps += fps
    # increment frame count
    frame_count += 1
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there's detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = np.round(scores[scores >= detection_threshold],2)
        draw_boxes = boxes.copy()


        # draw the bounding boxes and write the class name on top of it
        for j,box in enumerate(draw_boxes):
            cv2.rectangle(cv2_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          color_inference, 2)
            cv2.putText(img=cv2_image, text="Crater",
                        org=(int(box[0]), int(box[1] - 5)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale= 0.3,color= color_inference,
                        thickness=1, lineType=cv2.LINE_AA)
            cv2.putText(img=cv2_image, text=str(scores[j]),
                        org=(int(box[0]), int(box[1] + 8)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale= 0.3,color= color_inference,
                        thickness=1, lineType=cv2.LINE_AA)
            
        # add boxes for labels
        for box in data[1][0]['boxes']:
            cv2.rectangle(cv2_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          color_label, 2)
            cv2.putText(img=cv2_image, text="Label",
                        org=(int(box[0]), int(box[1] - 5)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale= 0.3,color= color_label,
                        thickness=1, lineType=cv2.LINE_AA)           
            

        # set size
        plt.figure(figsize=(10,10))
        plt.axis("off")

        # convert color from CV2 BGR back to RGB
        plt_image = cv2.cvtColor(cv2_image/255.0, cv2.COLOR_BGR2RGB)
        plt.imshow(plt_image)
        plt.show()
        cv2.imwrite(f"./results/{image_name}.jpg", cv2_image)
    print(f"Image {i + 1} done...")
    print('-' * 50)
print('TEST PREDICTIONS COMPLETE')

avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")








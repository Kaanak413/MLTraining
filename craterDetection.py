import os
from os import listdir
import sys
import json
import datetime

# import advance libraries
from xml.etree import ElementTree
import skimage.draw
import cv2
import imgaug

# import mask rcnn libraries
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn import visualize

# import matplotlib library
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import numpy libraries
import numpy as np
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean

# import keras libraries
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


class CraterDataset(Dataset):
   def load_dataset(self, dataset_dir):
        
        self.add_class("dataset", 1, "Crater")
        
        # we concatenate the dataset_dir with /images and /annots
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/labels/'
        for filename in listdir(images_dir):
            
            # extract image id
            image_id = filename[:-4] # used to skip last 4 chars which is '.jpg'
            

           
            
            # declaring image path and annotations path
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.txt'
            
            # using add_image function we pass image_id, image_path and ann_path so that the current
            # image is added to the dataset for training or testing
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
   
   def yolo_to_pixels(self,image_width, image_height, box):
     x, y, w, h = box
     xmin = int((x - w / 2) * image_width)
     xmax = int((x + w / 2) * image_width)
     ymin = int((y - h / 2) * image_height)
     ymax = int((y + h / 2) * image_height)
     return xmin, ymin, xmax, ymax

   def parse_labels_from_file(self,file_path):
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            label_parts = line.strip().split()
            if len(label_parts) == 5:
                class_id, x, y, w, h = map(float, label_parts)
                labels.append((int(class_id), x, y, w, h))
    return labels

   def extract_boxes(self, filename):
    boxes = list()
    img_width = self.load_image(0).shape[0]
    img_height = self.load_image(0).shape[1]
    labels = self.parse_labels_from_file(filename)
    for label in labels:
      class_id, x, y, w, h = label    
      xmin, ymin, xmax, ymax = self.yolo_to_pixels(img_width, img_height, (x, y, w, h))  
      tempList = [xmin, ymin, xmax, ymax]
      boxes.append(tempList)

    return boxes,img_width,img_height

   def load_mask(self, image_id):
        
        # info points to the current image_id
        info = self.image_info[image_id]
        
        path = info['annotation']
        
        boxes, w, h = self.extract_boxes(path)
        
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        
        class_ids = list()
        

        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('Crater'))
        
        # return masks and class_ids as array
        return masks, asarray(class_ids, dtype='int32')
    
    # this functions takes the image_id and returns the path of the image
   def image_reference(self, image_id):
    info = self.image_info[image_id]
    return info['path']    

class CraterConfig(Config):
    NAME = "crater_cfg"
    
    # crater class + background class
    NUM_CLASSES = 1 + 1
    
    GPU_COUNT = 1

    IMAGES_PER_GPU = 2
    # steps per epoch and minimum confidence
    STEPS_PER_EPOCH = 100
    
    # learning rate and momentum
    LEARNING_RATE=0.002
    
    # regularization penalty
    WEIGHT_DECAY = 0.0001
    
class InferConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    NAME = "crater"

# Function to convert label coordinates from YOLO format to pixel values

def putMasks(image,mask):
   
   for rois in mask:
       start_point = (rois[1], rois[0])
       end_point = (rois[3], rois[2]) 
       color = (255, 0, 0) 
       thickness = 2
       image = cv2.rectangle(image, start_point, end_point, color, thickness) 
   
   cv2.imshow("xd", image)
   cv2.waitKey(0)  


train_set = CraterDataset()
train_set.load_dataset('LU3M6TGT_yolo_format/train')
train_set.prepare()

test_set = CraterDataset()
test_set.load_dataset('LU3M6TGT_yolo_format/valid')
test_set.prepare()

config = CraterConfig()
Inferconfig = InferConfig()

model = MaskRCNN(mode='training', model_dir='./', config=config)
modelInference =  MaskRCNN(mode='inference', model_dir='./', config=Inferconfig)
weights_path = 'mask_rcnn_coco.h5'
mTrain = False
if(mTrain):
   weights_path = 'mask_rcnn_coco.h5'
   model.load_weights(weights_path, 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
   model.train(train_set, 
            test_set, 
            learning_rate=config.LEARNING_RATE, 
            epochs=5, 
            layers='heads')
else:
    weights_path = modelInference.find_last()
    modelInference.load_weights(weights_path, 
                   by_name=True, 
                   )
    image = skimage.io.imread("LU3M6TGT_yolo_format/valid/images/-0.28360998131250864,1.0882708585247933,13.17467296911068,14.546553808948.png")
    imageCV = cv2.imread("LU3M6TGT_yolo_format/valid/images/-0.28360998131250864,1.0882708585247933,13.17467296911068,14.546553808948.png")
    results = modelInference.detect([image],verbose=1)
    class_names = ['BG', 'Crater']
    r = results[0]
    putMasks(imageCV, r['rois'])
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names,  r['scores'])
    
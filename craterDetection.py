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

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import mrcnn.visualize
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

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

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
            
            image = skimage.io.imread(img_path)
            height, width = image.shape[:2]

            # using add_image function we pass image_id, image_path and ann_path so that the current
            # image is added to the dataset for training or testing
            self.add_image('dataset', image_id=image_id, path=img_path,width=width,height=height,annotation=ann_path)
   
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

   def extract_boxes(self, filename,height,width):
    boxes = list()
    img_width = width
    img_height = height
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
        height = info['height']
        width = info['width']
        boxes, w, h = self.extract_boxes(path,height,width)
        masks = zeros([height, width, len(boxes)], dtype=np.uint8)
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
    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.9
    IMAGE_MIN_DIM = 416
    IMAGE_MAX_DIM = 640
    
    
class InferConfig(CraterConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    

# Function to convert label coordinates from YOLO format to pixel values


def detect_and_get_masks(model, image_path):
    
    # Read image
    image = skimage.io.imread(image_path)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    # Color splash
    semantic_masks, instance_masks = putMasks(image, r['masks'], r['class_ids'])

    plt.subplot(1, 2, 1)
    plt.title('rgb')
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title('masks')
    plt.imshow(instance_masks)
    plt.show()
    
    # Save output
    file_name = "mask_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, instance_masks)
    print("Saved to ", file_name)
  
train_set = CraterDataset()
train_set.load_dataset('LU3M6TGT_yolo_format/train')
train_set.prepare()

test_set = CraterDataset()
test_set.load_dataset('LU3M6TGT_yolo_format/valid')
test_set.prepare()

def putMasks(image,mask,givenCoordinates):
   for rois in mask:
       start_point = (rois[1], rois[0])
       end_point = (rois[3], rois[2]) 
       color = (255, 0, 0) 
       thickness = 3
       image = cv2.rectangle(image, start_point, end_point, color, thickness) 
   for label in givenCoordinates:
      class_id, x, y, w, h = label

      xmin, ymin, xmax, ymax = train_set.yolo_to_pixels(image.shape[0], image.shape[1], (x, y, w, h))
      start_point = (xmin, ymin)
      end_point = (xmax, ymax) 
      color = (0, 0, 255) 
      thickness = 1
      image = cv2.rectangle(image, start_point, end_point, color, thickness) 

   cv2.imshow("Image", image)
   cv2.waitKey(0) 

config = CraterConfig()
Inferconfig = InferConfig()

model = MaskRCNN(mode='training', model_dir='./', config=config)
modelInference =  modellib.MaskRCNN(mode='inference', model_dir='./', config=Inferconfig)
weights_path = 'mask_rcnn_coco.h5'
mTrain = False
if(mTrain):
   weights_path = model.find_last()
   model.load_weights(weights_path, 
                   by_name=True)
   model.train(train_set, 
            test_set, 
            learning_rate=config.LEARNING_RATE, 
            epochs=30, 
            layers='heads')
else:
    weights_path = 'mask_rcnn_crater_cfg_0005.h5'
    modelInference.load_weights(weights_path, 
                   by_name=True, 
                   )

    image = skimage.io.imread("LU3M6TGT_yolo_format/valid/images/-0.28360998131250864,1.0882708585247933,13.17467296911068,14.546553808948.png")
    imageCV = cv2.imread("LU3M6TGT_yolo_format/valid/images/-0.28360998131250864,1.0882708585247933,13.17467296911068,14.546553808948.png")
    results = modelInference.detect([image],verbose=1)
    class_names = ['BG', 'Crater']
    r = results[0]
    mrcnn.visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            test_set.class_names, r['scores'], ax=get_ax())
#     mrcnn = modelInference.run_graph([image], [
#     ("detections", modelInference.keras_model.get_layer("mrcnn_detection").output),
#     ("masks", modelInference.keras_model.get_layer("mrcnn_mask").output),
# ])

# # Get detection class IDs. Trim zero padding.
#     det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
#     det_count = np.where(det_class_ids == 0)[0][0]
#     det_class_ids = det_class_ids[:det_count]

#     print("{} detections: {}".format(
#     det_count, np.array(test_set.class_names)[det_class_ids]))
#     det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
#     det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c] 
#                               for i, c in enumerate(det_class_ids)])
#     det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)
#                       for i, m in enumerate(det_mask_specific)])
#     log("det_mask_specific", det_mask_specific)
#     log("det_masks", det_masks)
#     display_images(det_mask_specific[:4] * 255, cmap="Blues", interpolation="none")
#     display_images(det_masks[:4] * 255, cmap="Blues", interpolation="none")
    # detect_and_get_masks(modelInference,img_path)
    # path= "LU3M6TGT_yolo_format/valid/labels/-0.28360998131250864,1.0882708585247933,13.17467296911068,14.546553808948.txt"
    # putMasks(imageCV, r['rois'],test_set.parse_labels_from_file(path))
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names,  r['scores'])
    image_ids = np.random.choice(test_set.image_ids, 750)
    APs = []
    for image_id in image_ids:
    # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(test_set, Inferconfig,
                               image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, Inferconfig), 0)
    # Run object detection
        results = modelInference.detect([image], verbose=0)
        r = results[0]
    # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
    
    print("mAP: ", np.mean(APs))
    
    
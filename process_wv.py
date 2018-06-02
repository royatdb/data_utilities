"""
Copyright 2018 Defense Innovation Unit Experimental
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


from PIL import Image
import tensorflow as tf
import io
import glob
from tqdm import tqdm
import numpy as np
import logging
import argparse
import os
import json
import wv_util as wv
import tfr_util as tfr
import aug_util as aug
import csv

"""
  A script that processes xView imagery. 
  Args:
      image_folder: A folder path to the directory storing xView .tif files
        ie ("xView_data/")

      json_filepath: A file path to the GEOJSON ground truth file
        ie ("xView_gt.geojson")

      test_percent (-t): The percentage of input images to use for test set

      suffix (-s): The suffix for output TFRecord files.  Default suffix 't1' will output
        xview_train_t1.record and xview_test_t1.record

      augment (-a): A boolean value of whether or not to use augmentation

  Outputs:
    Writes two files to the current directory containing training and test data in
        TFRecord format ('xview_train_SUFFIX.record' and 'xview_test_SUFFIX.record')
"""


def get_images_from_filename_array(coords,chips,classes,folder_names,res=(250,250)):
    """
    Gathers and chips all images within a given folder at a given resolution.

    Args:
        coords: an array of bounding box coordinates
        chips: an array of filenames that each coord/class belongs to.
        classes: an array of classes for each bounding box
        folder_names: a list of folder names containing images
        res: an (X,Y) tuple where (X,Y) are (width,height) of each chip respectively

    Output:
        images, boxes, classes arrays containing chipped images, bounding boxes, and classes, respectively.
    """

    images =[]
    boxes = []
    clses = []

    k = 0
    bi = 0   
    
    for folder in folder_names:
        fnames = glob.glob(folder + "*.tif")
        fnames.sort()
        for fname in tqdm(fnames):
            #Needs to be "X.tif" ie ("5.tif")
            name = fname.split("\\")[-1]
            arr = wv.get_image(fname)
            
            img,box,cls = wv.chip_image(arr,coords[chips==name],classes[chips==name],res)

            for im in img:
                images.append(im)
            for b in box:
                boxes.append(b)
            for c in cls:
                clses.append(cls)
            k = k + 1
            
    return images, boxes, clses

def shuffle_images_and_boxes_classes(im,box,cls):
    """
    Shuffles images, boxes, and classes, while keeping relative matching indices

    Args:
        im: an array of images
        box: an array of bounding box coordinates ([xmin,ymin,xmax,ymax])
        cls: an array of classes

    Output:
        Shuffle image, boxes, and classes arrays, respectively
    """
    assert len(im) == len(box)
    assert len(box) == len(cls)
    
    perm = np.random.permutation(len(im))
    out_b = {}
    out_c = {}
    
    k = 0 
    for ind in perm:
        out_b[k] = box[ind]
        out_c[k] = cls[ind]
        k = k + 1
    return im[perm], out_b, out_c

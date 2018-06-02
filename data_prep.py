import tensorflow as tf
import io
import glob
from tqdm import tqdm
import numpy as np
import argparse
import os
import json
import csv
from PIL import Image, ImageDraw
import skimage.filters as filters

from wv_util import *
from tfr_util import *
from aug_util import *
from process_wv import *


image_folder = "/Users/debajyotiroy/Downloads/xview_union_images/"
json_filepath = "/Users/debajyotiroy/Downloads/xView_train.geojson"
test_percent=0.0
suffix="db"
AUGMENT = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#resolutions should be largest -> smallest.  We take the number of chips in the largest resolution and make
#sure all future resolutions have less than 1.5times that number of images to prevent chip size imbalance.
#res = [(500,500),(400,400),(300,300),(200,200)]
res = [(300,300)]

SAVE_IMAGES = False
images = {}
boxes = {}
train_chips = 0
test_chips = 0

#Parameters
max_chips_per_res = 100000
train_writer = tf.python_io.TFRecordWriter("xview_train_%s.record" % suffix)
test_writer = tf.python_io.TFRecordWriter("xview_test_%s.record" % suffix)

coords,chips,classes = wv.get_labels(json_filepath)

for res_ind, it in enumerate(res):
    tot_box = 0
    logging.info("Res: %s" % str(it))
    ind_chips = 0

    fnames = glob.glob(image_folder + "*.tif")
    fnames.sort()

    for fname in tqdm(fnames):
        #Needs to be "X.tif", ie ("5.tif")
        #Be careful!! Depending on OS you may need to change from '/' to '\\'.  Use '/' for UNIX and '\\' for windows
        name = fname.split("/")[-1]
        arr = wv.get_image(fname)

        im,box,classes_final = wv.chip_image(arr,coords[chips==name],classes[chips==name],it)

        #Shuffle images & boxes all at once. Comment out the line below if you don't want to shuffle images
        im,box,classes_final = shuffle_images_and_boxes_classes(im,box,classes_final)
        split_ind = int(im.shape[0] * test_percent)

        for idx, image in enumerate(im):
            tf_example = tfr.to_tf_example(image,box[idx],classes_final[idx])

            #Check to make sure that the TF_Example has valid bounding boxes.  
            #If there are no valid bounding boxes, then don't save the image to the TFRecord.
            float_list_value = tf_example.features.feature['image/object/bbox/xmin'].float_list.value
            
            if (ind_chips < max_chips_per_res and np.array(float_list_value).any()):
                tot_box+=np.array(float_list_value).shape[0]
                
                if idx < split_ind:
                    test_writer.write(tf_example.SerializeToString())
                    test_chips+=1
                else:
                    train_writer.write(tf_example.SerializeToString())
                    train_chips += 1
 
                ind_chips +=1

                #Make augmentation probability proportional to chip size.  Lower chip size = less chance.
                #This makes the chip-size imbalance less severe.
                prob = np.random.randint(0,np.max(res))
                #for 200x200: p(augment) = 200/500 ; for 300x300: p(augment) = 300/500 ...

                if AUGMENT and prob < it[0]:
                    
                    for extra in range(3):
                        center = np.array([int(image.shape[0]/2),int(image.shape[1]/2)])
                        deg = np.random.randint(-10,10)
                        #deg = np.random.normal()*30
                        newimg = aug.salt_and_pepper(aug.gaussian_blur(image))

                        #.3 probability for each of shifting vs rotating vs shift(rotate(image))
                        p = np.random.randint(0,3)
                        if p == 0:
                            newimg,nb = aug.shift_image(newimg,box[idx])
                        elif p == 1:
                            newimg,nb = aug.rotate_image_and_boxes(newimg,deg,center,box[idx])
                        elif p == 2:
                            newimg,nb = aug.rotate_image_and_boxes(newimg,deg,center,box[idx])
                            newimg,nb = aug.shift_image(newimg,nb)
                            

                        newimg = (newimg).astype(np.uint8)

                        if idx%1000 == 0 and SAVE_IMAGES:
                            Image.fromarray(newimg).save('process/img_%s_%s_%s.png'%(name,extra,it[0]))

                        if len(nb) > 0:
                            tf_example = tfr.to_tf_example(newimg,nb,classes_final[idx])

                            #Don't count augmented chips for chip indices
                            if idx < split_ind:
                                test_writer.write(tf_example.SerializeToString())
                                test_chips += 1
                            else:
                                train_writer.write(tf_example.SerializeToString())
                                train_chips+=1
                        else:
                            if SAVE_IMAGES:
                                aug.draw_bboxes(newimg,nb).save('process/img_nobox_%s_%s_%s.png'%(name,extra,it[0]))
    if res_ind == 0:
        max_chips_per_res = int(ind_chips * 1.5)
        logging.info("Max chips per resolution: %s " % max_chips_per_res)

    logging.info("Tot Box: %d" % tot_box)
    logging.info("Chips: %d" % ind_chips)

logging.info("saved: %d train chips" % train_chips)
logging.info("saved: %d test chips" % test_chips)
train_writer.close()
test_writer.close() 
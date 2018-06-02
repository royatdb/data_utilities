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


image_folder = "/Users/debajyotiroy/Downloads/train_images/"
json_filepath = "/Users/debajyotiroy/Downloads/xView_train.geojson"
test_percent=0.0
suffix="db"
AUGMENT = True

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

coords,chips,classes = get_labels(json_filepath)

for res_ind, it in enumerate(res):
    tot_box = 0
    print("Res: %s" % str(it))
    ind_chips = 0

    fnames = glob.glob(image_folder + "*.tif")
    fnames.sort()

    for fname in tqdm(fnames):
        name = fname.split("/")[-1]
        arr = get_image(fname)

        im,box,classes_final = chip_image(arr,coords[chips==name],classes[chips==name],it)

        #Shuffle images & boxes all at once. Comment out the line below if you don't want to shuffle images
        im,box,classes_final = shuffle_images_and_boxes_classes(im,box,classes_final)
        split_ind = int(im.shape[0] * test_percent)

        for idx, image in enumerate(im):
            tf_example = to_tf_example(image,box[idx],classes_final[idx])

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
                        newimg = salt_and_pepper(gaussian_blur(image))

                        #.3 probability for each of shifting vs rotating vs shift(rotate(image))
                        p = np.random.randint(0,3)
                        if p == 0:
                            newimg,nb = shift_image(newimg,box[idx])
                        elif p == 1:
                            newimg,nb = rotate_image_and_boxes(newimg,deg,center,box[idx])
                        elif p == 2:
                            newimg,nb = rotate_image_and_boxes(newimg,deg,center,box[idx])
                            newimg,nb = shift_image(newimg,nb)


                        newimg = (newimg).astype(np.uint8)

                        if idx%1000 == 0 and SAVE_IMAGES:
                            Image.fromarray(newimg).save('process/img_%s_%s_%s.png'%(name,extra,it[0]))

                        if len(nb) > 0:
                            tf_example = to_tf_example(newimg,nb,classes_final[idx])

                            #Don't count augmented chips for chip indices
                            if idx < split_ind:
                                test_writer.write(tf_example.SerializeToString())
                                test_chips += 1
                            else:
                                train_writer.write(tf_example.SerializeToString())
                                train_chips+=1
                        else:
                            if SAVE_IMAGES:
                                draw_bboxes(newimg,nb).save('process/img_nobox_%s_%s_%s.png'%(name,extra,it[0]))
    if res_ind == 0:
        max_chips_per_res = int(ind_chips * 1.5)
        print("Max chips per resolution: %s " % max_chips_per_res)

    print("Tot Box: %d" % tot_box)
    print("Chips: %d" % ind_chips)

print("saved: %d train chips" % train_chips)
print("saved: %d test chips" % test_chips)
train_writer.close()
test_writer.close() 

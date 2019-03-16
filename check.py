# -*- coding: utf-8 -*-
import copy

import numpy as np
import tensorflow as tf
from PIL import Image
import time

from tensorflow.contrib.optimizer_v2.gradient_descent import GradientDescentOptimizer
from tensorflow.python.keras.losses import mse

import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
    load_graph, letter_box_image, generate_ground_truth

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

size = 416
frozen_model = 'frozen_darknet_yolov3_model.pb'
# frozen_model = None
tiny = False
data_format = 'NHWC'
ckpt_file = 'saved_model/checkpoint'


conf_threshold = 0.5
iou_threshold = 0.4

def main(argv=None):

    img = Image.open('out/images/19.png')
    # img = Image.open('city.png')
    img_resized = letter_box_image(img, size, size, 128)
    img_resized = img_resized.astype(np.float32)
    classes = load_coco_names('coco.names')

    if frozen_model:

        t0 = time.time()
        frozenGraph = load_graph(frozen_model)
        print("Loaded graph in {:.2f}s".format(time.time()-t0))

        boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)

        with tf.Session(graph=frozenGraph) as sess:
            t0 = time.time()
            detected_boxes = sess.run(
                boxes, feed_dict={inputs: [img_resized]})

    else:
        if tiny:
            model = yolo_v3_tiny.yolo_v3_tiny
        else:
            model = yolo_v3.yolo_v3

        boxes, inputs = get_boxes_and_inputs(model, len(classes), size, data_format)

        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

        with tf.Session() as sess:
            t0 = time.time()
            saver.restore(sess, ckpt_file)
            print('Model restored in {:.2f}s'.format(time.time()-t0))

            t0 = time.time()
            detected_boxes = sess.run(
                boxes, feed_dict={inputs: [img_resized]})

    filtered_boxes = non_max_suppression(detected_boxes,
                                         confidence_threshold=conf_threshold,
                                         iou_threshold=iou_threshold)
    print("Predictions found in {:.2f}s".format(time.time() - t0))

    draw_boxes(filtered_boxes, img, classes, (size, size), True)
    img.save('out_check.png')


if __name__ == '__main__':
    tf.app.run()

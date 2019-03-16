# -*- coding: utf-8 -*-
import copy

import numpy as np
import tensorflow as tf
from PIL import Image
import time

from scipy.ndimage import gaussian_filter
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

num_iterations = 20
step_size = 1.0

conf_threshold = 0.5
iou_threshold = 0.4

def main(argv=None):

    img = Image.open('city.png')
    img_resized = letter_box_image(img, size, size, 128)
    img_resized = img_resized.astype(np.float32)
    classes = load_coco_names('coco.names')

    fake_boxes = {
        2: [(np.array([300, 200, 370, 250]), 1.)]
    }
    generated_boxes, g_indices = generate_ground_truth(fake_boxes, size, 0.4)
    draw_boxes(copy.deepcopy(generated_boxes), img, classes, (size, size), True)
    draw_boxes(copy.deepcopy(fake_boxes), img, classes, (size, size), True)
    # draw_boxes(filtered_boxes, img, classes, (size, size), True)
    img.save('out_fakeboxes.jpg')

    mask = np.zeros([1, 10647])
    for cls, indices in g_indices.items():
        mask[0, indices] = 1

    gt_tensor = np.zeros([1, 10647, 4 + 1 + len(classes)])
    for cls, boxes in generated_boxes.items():
        for i, box in enumerate(boxes):
            class_mask = np.zeros([len(classes)])
            class_mask[cls] = 1
            gt_row = [*np.asarray(box[0]), 1., *class_mask]
            gt_tensor[0, g_indices[cls][i]] = gt_row

    if frozen_model:

        t0 = time.time()
        frozenGraph = load_graph(frozen_model)
        print("Loaded graph in {:.2f}s".format(time.time()-t0))

        boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)

        with frozenGraph.as_default():
            fake_gt = tf.constant(gt_tensor, dtype=tf.float32)
            mask_tensor = tf.constant(mask, dtype=tf.float32)
            fake_loss = mse(fake_gt, boxes) * mask_tensor
            fake_loss = tf.reduce_mean(fake_loss, axis=-1)

            grad_op = tf.gradients(fake_loss, inputs)


        with tf.Session(graph=frozenGraph) as sess:
            t0 = time.time()
            for iters in range(num_iterations):
                grads = sess.run(
                    grad_op, feed_dict={inputs: [img_resized]})

                grad = grads[0][0]
                sigma = (iters * 4.0) / num_iterations + 0.5
                grad_smooth1 = gaussian_filter(grad, sigma=sigma)
                grad_smooth2 = gaussian_filter(grad, sigma=sigma * 2)
                grad_smooth3 = gaussian_filter(grad, sigma=sigma * 0.5)
                grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)

                step_size_scaled = step_size / (np.std(grad) + 1e-8)

                # Update the image by following the gradient.
                mod = grad * step_size_scaled

                grad_img = Image.fromarray(np.uint8(mod + 128))
                grad_img.save('out/grads/{}.png'.format(iters))

                img_resized = np.clip(img_resized - mod, 0, 255)
                new_img = Image.fromarray(np.uint8(img_resized))
                new_img.save('out/images/{}.png'.format(iters))


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


if __name__ == '__main__':
    tf.app.run()

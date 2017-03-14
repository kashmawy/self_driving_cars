import tensorflow as tf
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, help='movie or picture')
parser.add_argument('--input', type=str, help='the input')
parser.add_argument('--output', type=str, nargs='?', help='the output')

class YOLO:
    alpha = 0.1
    weights_file = 'weights/YOLO_tiny.ckpt'
    input_size = (448, 448)
    threshold = 0.20
    iou_threshold = 0.5
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"] # 20 classes

    def __init__(self, input, output, is_movie):
        self.build_model()

        if is_movie:
            self.convert_video(input, output)
        else:
            output_image = self.pipeline(cv2.imread(input))
            self.show(output_image)

    def convert_video(self, invideo, outvideo):
        input_clip = VideoFileClip(invideo)
        output_clip = input_clip.fl_image(self.pipeline)
        output_clip.write_videofile(outvideo, audio=False)

    def pipeline(self, image):
        output = self.predict_from_image(image)
        boxes = self.interpret_results(output)
        result = self.draw_boxes(image, boxes)
        image_converted = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return image_converted

    def show(self, image):
        plt.imshow(image)
        plt.show()

    def draw_boxes(self, image, boxes):
        image_copy = image.copy()
        for i in range(len(boxes)):
            x = int(boxes[i][1])
            y = int(boxes[i][2])
            w = int(boxes[i][3]) // 2
            h = int(boxes[i][4]) // 2

            cv2.rectangle(image_copy, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(image_copy, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
            cv2.putText(image_copy, boxes[i][0] + ' : %.2f' % boxes[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image_copy

    def interpret_results(self, output):
        probabilities = np.zeros((7, 7, 2, 20))
        class_probabilities = np.reshape(output[0:980], (7, 7, 20))
        scales = np.reshape(output[980:1078], (7, 7, 2))
        boxes = np.reshape(output[1078:], (7, 7, 2, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)), (1, 2, 0))

        boxes[:,:,:,0] += offset
        boxes[:,:,:,1] += np.transpose(offset, (1, 0, 2))
        boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
        boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2], boxes[:,:,:,2])
        boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3], boxes[:,:,:,3])

        boxes[:,:,:,0] *= self.image_width
        boxes[:,:,:,1] *= self.image_height
        boxes[:,:,:,2] *= self.image_width
        boxes[:,:,:,3] *= self.image_height

        for i in range(2):
            for j in range(20):
                probabilities[:,:,i,j] = np.multiply(class_probabilities[:,:,j], scales[:,:,i])

        filter_mat_probabilities = np.array(probabilities>=self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probabilities)
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probabilities[filter_mat_probabilities]
        class_num_filtered = np.argmax(filter_mat_probabilities, axis=3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        class_num_filtered = class_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0: continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.intersection_over_union(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered>0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        class_num_filtered = class_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([
                self.classes[class_num_filtered[i]],
                boxes_filtered[i][0],
                boxes_filtered[i][1],
                boxes_filtered[i][2],
                boxes_filtered[i][3],
                probs_filtered[i]
            ])

        return result

    def intersection_over_union(self, box1, box2):
        tb = min(
            box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]
        ) - max(
            box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2]
        )

        lr = min(
            box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]
        ) - max(
            box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3]
        )

        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr

        box1Area = box1[2] * box1[3]
        box2Area = box2[2] * box2[3]

        union = float(box1Area + box2Area - intersection)

        return intersection / union

    def predict_from_image(self, image):
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]

        image_resized = cv2.resize(image, self.input_size)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_np = np.asarray(image_rgb)
        inputs = np.zeros((1, 448, 448, 3),dtype='float32')
        inputs[0] = ((image_np / 255.0) * 2.0) - 1.0
        input_dictionary = {self.x: inputs}
        output = self.sess.run(self.fc_19, feed_dict=input_dictionary)
        return output[0]


    def build_model(self):
        print("Building YOLO model")
        self.x = tf.placeholder('float32', [None, self.input_size[0], self.input_size[1], 3])

        # First Layer
        self.conv_1 = self.conv_layer(1, self.x, 16, 3, 1)
        self.pool_2 = self.pool_layer(2, self.conv_1, 2, 2)

        # Second Layer
        self.conv_3 = self.conv_layer(3, self.pool_2, 32, 3, 1)
        self.pool_4 = self.pool_layer(4, self.conv_3, 2, 2)

        # Third Layer
        self.conv_5 = self.conv_layer(5, self.pool_4, 64, 3, 1)
        self.pool_6 = self.pool_layer(6, self.conv_5, 2, 2)

        # Fourth Layer
        self.conv_7 = self.conv_layer(7, self.pool_6, 128, 3, 1)
        self.pool_8 = self.pool_layer(8, self.conv_7, 2, 2)

        # Fifth Layer
        self.conv_9 = self.conv_layer(9, self.pool_8, 256, 3, 1)
        self.pool_10 = self.pool_layer(10, self.conv_9, 2, 2)

        # Sixth Layer
        self.conv_11 = self.conv_layer(11, self.pool_10, 512, 3, 1)
        self.pool_12 = self.pool_layer(12, self.conv_11, 2, 2)

        # Eights Latyer
        self.conv_13 = self.conv_layer(13, self.pool_12, 1024, 3, 1)
        self.conv_14 = self.conv_layer(14, self.conv_13, 1024, 3, 1)
        self.conv_15 = self.conv_layer(15, self.conv_14, 1024, 3, 1)

        # Ninth Layer
        self.fc_16 = self.fc_layer(16, self.conv_15, 256, flat=True, linear=False)
        self.fc_17 = self.fc_layer(17, self.fc_16, 4096, flat=False, linear=False)
        self.fc_19 = self.fc_layer(19, self.fc_17, 1470, flat=False, linear=True)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def conv_layer(self, layer_id, inputs, filters, size, stride):
        channels = inputs.get_shape()[3]

        weight = tf.Variable(tf.truncated_normal([size, size, int(channels), filters], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))

        pad_size = size//2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        inputs_pad = tf.pad(inputs, pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID', name=str(layer_id)+'_conv')
        conv_biased = tf.add(conv, biases, name=str(layer_id) + '_conv_biased')

        return tf.maximum(self.alpha * conv_biased, conv_biased, name=str(layer_id) + '_leaky_relu')

    def pool_layer(self, layer_id, inputs, size, stride):
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME', name=str(layer_id) + '_pool')

    def fc_layer(self, layer_id, inputs, hiddens, flat=False, linear=False):
        input_shape = inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1]*input_shape[2]*input_shape[3]
            inputs_tranposed = tf.transpose(inputs, (0, 3, 1, 2))
            inputs_processed = tf.reshape(inputs_tranposed, [-1, dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs

        weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))

        if linear:
            return tf.add(tf.matmul(inputs_processed, weight), biases, name=str(layer_id) + '_fc')

        ip = tf.add(tf.matmul(inputs_processed, weight), biases)
        return tf.maximum(self.alpha * ip, ip, name=str(layer_id) + '_fc')

if __name__=='__main__':
    result = parser.parse_args()

    if result.type != 'movie' and result.type != 'picture':
        print("Type can only be movie or picture")
        exit()
    if result.type == 'movie' and not result.output:
        print("output must be set for a movie")
        exit()

    YOLO(result.input, result.output, result.type == 'movie')

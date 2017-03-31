import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
import argparse
from networks.factory import get_network
from fast_rcnn.nms_wrapper import nms
from model import VGGNet
import numpy as np
import cv2
import matplotlib.pyplot as plt

PIXEL_MEAN = np.array([[[102.9801, 115.9465, 122.7717]]])
SCALES = (600,)
MAX_SIZE = 1000
CLASSES = (
    '__background__',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
)


def images_list_to_blob(images):
    max_shape = np.array([im.shape for im in images]).max(axis=0)
    images_count = len(images)
    blob = np.zeros((images_count, max_shape[0], max_shape[1], 3), dtype=np.float32)
    for i in xrange(images_count):
        image = images[i]
        blob[i, 0:image.shape[0], 0:image.shape[1], :] = image

    return blob

def get_input(image):
    image_copy = image.astype(np.float32, copy=True)
    image_copy -= PIXEL_MEAN

    image_shape = image_copy.shape
    image_shape_min = np.min(image_shape[0:2])
    image_shape_max = np.max(image_shape[0:2])

    processed_images = []
    image_scale_factors = []

    for target_size in SCALES:
        image_scale = float(target_size) / float(image_shape_min)
        if np.round(image_scale * image_shape_max) > MAX_SIZE:
            image_scale = float(MAX_SIZE) / float(image_shape_max)
        image = cv2.resize(image_copy, None, None, fx=image_scale, fy=image_scale, interpolation=cv2.INTER_LINEAR)
        image_scale_factors.append(image_scale)
        processed_images.append(image)

    blob = images_list_to_blob(processed_images)

    blobs = {
        'data': blob,
        'rois': None
    }
    im_scale_factors = np.array(image_scale_factors)

    return blobs, im_scale_factors


def detect(session, network, image):
    blobs_data, image_scales = get_input(image)
    image_blob = blobs_data['data']
    blobs_data['im_info'] = np.array(
        [[image_blob.shape[1], image_blob.shape[2], image_scales[0]]],
        dtype=np.float32
    )

    feed_dict = {
        network.data: blobs_data['data'],
        network.im_info: blobs_data['im_info'],
        network.keep_prob: 1.0
    }

    cls_score, cls_prob, bbox_pred, rois = session.run(
        [network.get_output('cls_score'), network.get_output('cls_prob'), network.get_output('bbox_pred'), network.get_output('rois')],
        feed_dict=feed_dict
    )

    boxes = rois[:, 1:5] / image_scales[0]

    pred_boxes = bbox_transform_inv(boxes, bbox_pred)
    pred_boxes = clip_boxes(pred_boxes, image.shape)

    return cls_score, pred_boxes

def visualize(image, scores, boxes):
    # Visualize detections for each class
    image = image[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, aspect='equal')

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(cls, dets, ax, thresh=CONF_THRESH)

    plt.show()


def vis_detections(class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--image', dest='image_path', help='The image to load', type=str)
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))

    session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    network = VGGNet()
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(session, args.model)

    image = cv2.imread(args.image_path)
    scores, boxes = detect(session, network, image)
    visualize(image, scores, boxes)
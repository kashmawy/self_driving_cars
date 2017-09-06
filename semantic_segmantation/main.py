import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    vgg_input_tensor_name = 'image_input:0'
    vgg_input_tensor = sess.graph.get_tensor_by_name(vgg_input_tensor_name)

    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_keep_prob_tensor = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer3_out_tensor = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)

    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer4_out_tensor = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)

    vgg_layer7_out_tensor_name = 'layer7_out:0'
    vgg_layer7_out_tensor = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return (
        vgg_input_tensor,
        vgg_keep_prob_tensor,
        vgg_layer3_out_tensor,
        vgg_layer4_out_tensor,
        vgg_layer7_out_tensor
    )
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    fc_layer = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    layer7_upsample = tf.layers.conv2d_transpose(
        fc_layer,
        num_classes,
        4, 2,
        'SAME',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
    )

    layer4_skip_conv = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    layer4_skip_connection = tf.add(layer7_upsample, layer4_skip_conv)
    layer4_upsample = tf.layers.conv2d_transpose(
        layer4_skip_connection,
        num_classes,
        4, 2,
        'SAME',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
    )

    layer3_skip_conv = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    layer3_skip_connection = tf.add(layer4_upsample, layer3_skip_conv)
    layer3_upsample = tf.layers.conv2d_transpose(
        layer3_skip_connection,
        num_classes,
        16, 8,
        'SAME',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
    )

    return layer3_upsample
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="Softmax"))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, labels, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    count = 0
    for epoch in range(epochs):
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run(
                [train_op, cross_entropy_loss],
                feed_dict={
                    input_image: image,
                    correct_label: label,
                    keep_prob: 0.70,
                    learning_rate: 0.0005
                }
            )
            print(count)
            if count % 2 == 0:
                print("Epoch {}/{}...".format(epoch, epochs),
                      "Training loss: {:.4f}...".format(loss))
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    batch_size = 10
    epochs = 10

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        output_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        learning_rate = tf.placeholder(dtype=tf.float32)
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))

        reshaped_logits, train_op, cross_entropy_loss = optimize(output_layer, correct_label, learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())
        train_nn(
            sess,
            epochs,
            batch_size,
            get_batches_fn,
            train_op,
            cross_entropy_loss,
            image_input,
            correct_label,
            keep_prob,
            learning_rate
        )

        saver = tf.train.Saver()
        saver.save(sess, 'model/model.ckpt')
        saver.export_meta_graph('model/final.meta')
        tf.train.write_graph(sess.graph_def, './model', 'final.pb', False)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, reshaped_logits, keep_prob, image_input)



if __name__ == '__main__':
    run()

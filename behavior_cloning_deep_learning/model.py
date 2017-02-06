import tensorflow as tf
import numpy as np
import csv
import random
from keras.optimizers import Adam
from keras.layers import (
    Flatten, Dense, Dropout, Convolution2D, Activation, BatchNormalization
)
from keras.models import Model, Sequential, model_from_json
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split
import json

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('save_file', 'output_model', "The model and weights file to save (.json and .h5)")
flags.DEFINE_string('driving_log', 'driving_log.csv', 'The driving log.')
flags.DEFINE_integer('epochs', 8, "The number of epochs.")
flags.DEFINE_integer('batch_size', 64, "The batch size.")
flags.DEFINE_integer('epoch_sample', 1000, 'The epoch sample.')
flags.DEFINE_float('lrate', 0.001, 'The learning rate')
flags.DEFINE_integer('validation_sample', 1000, 'The validation sample.')

CSV_CENTER_IMAGE_INDEX = 0
CSV_LEFT_IMAGE_INDEX = 1
CSV_RIGHT_IMAGE_INDEX = 2
CSV_STEERING_IMAGE_INDEX = 3
CSV_THROTTLE_IMAGE_INDEX = 4
CSV_BRAKE_IMAGE_INDEX = 5
CSV_SPEED_IMAGE_INDEX = 6


def nvidia_model(image):
    model = Sequential()
    model.add(BatchNormalization(axis=1, input_shape=image.shape))
    model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(24, 3, 3, border_mode='valid', subsample=(1, 2), activation='elu'))
    model.add(Convolution2D(36, 3, 3, border_mode='valid', activation='elu'))
    model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='elu'))
    model.add(Convolution2D(48, 2, 2, border_mode='valid', activation='elu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(Activation('elu'))
    model.add(Dense(10))
    model.add(Activation('elu'))
    model.add(Dense(1))
    model.summary()
    adam = Adam(lr=0.0001)
    model.compile(loss='mse',
                  optimizer=adam)
    return model


def load_csv(path):
    csv_rows = []
    with open(path, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            csv_rows.append(row)

    preprocess(csv_rows)

    csv_rows_main, csv_rows_test = train_test_split(csv_rows, test_size=0.1)
    csv_rows_train, csv_rows_val = train_test_split(csv_rows_main, test_size=0.1)
    return (csv_rows_train, csv_rows_val, csv_rows_test)


def normalize(imgs):
    """
    Normalize images between [-1, 1].
    """
    return (imgs / 255.0) - 0.5


def flip(image, steering):
    return (np.fliplr(image), -steering)


def crop(imgs):
    result = []
    for img in imgs:
        result_img = img[10: , :, :]
        result.append(result_img)

    return result


def resize(imgs, shape=(20, 64, 3)):
    """
    Resize images to shape.
    """
    height, width, channels = shape
    imgs_resized = np.empty([len(imgs), height, width, channels])
    for i, img in enumerate(imgs):
        imgs_resized[i] = imresize(img, shape)

    return imgs_resized


def preprocess_image(img):
    img = crop(img)
    img = resize(img)
    img = normalize(img)

    return img


def generator_from(csv_rows):
    while True:
        for i in range(0, len(csv_rows)):
            current_images = []
            current_angeles = []
            for j in range(0, FLAGS.batch_size):
                angle = float(csv_rows[i][3])

                current_images.append(imread(csv_rows[i][0].strip()).astype(np.float32))
                current_angeles.append(angle)

                if csv_rows[i][1] != '':
                    current_images.append(imread(csv_rows[i][1].strip()).astype(np.float32))
                    current_angeles.append(angle + .25)

                if csv_rows[i][2] != '':
                    current_images.append(imread(csv_rows[i][2].strip()).astype(np.float32))
                    current_angeles.append(angle - .25)

                (new_image, new_angle) = flip(imread(csv_rows[i][0]).astype(np.float32), angle)
                current_images.append(new_image)
                current_angeles.append(new_angle)

            current_images = preprocess_image(current_images)
            yield (current_images, current_angeles)


def get_image(path):
    with open(path, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            image = imread(rows[0]).astype(np.float32)
            image = image - np.mean(image)
            # csv.close(infile)
            return preprocess_image(np.array([image]))[0]
            # return image


def save_model(model):
    print("Saving model...")
    model.save_weights(FLAGS.weights_file)

    model_as_json = model.to_json()
    with open(FLAGS.model_file, "w") as model_file:
        model_file.write(model_as_json)
    print("Model saved.")


def save(model, prefix):
    """save model for future inspection and continuous training
    """
    model_file = prefix + ".json"
    weight_file = prefix + ".h5"
    json.dump(model.to_json(), open(model_file, "w"))
    model.save_weights(weight_file)
    print("Model saved.")
    return model


def restore(prefix):
    """restore a saved model
    """
    model_file = prefix + ".json"
    weight_file = prefix + ".h5"
    model = model_from_json(json.load(open(model_file)))
    model.load_weights(weight_file)
    print("Model loaded.")
    return model


def shuffle(csv_rows):
    print("Shuffled the data.")
    random.shuffle(csv_rows)
    return csv_rows


def preprocess(csv_rows):
    csv_rows = shuffle(csv_rows)
    return csv_rows


def main(_):
    (csv_rows_train, csv_rows_val, csv_rows_test) = load_csv(FLAGS.driving_log)

    image = get_image(FLAGS.driving_log)
    model = nvidia_model(image)

    model.fit_generator(
        generator=generator_from(csv_rows_train),
        samples_per_epoch=FLAGS.epoch_sample,
        nb_epoch=FLAGS.epochs,
        validation_data=generator_from(csv_rows_val),
        nb_val_samples=FLAGS.validation_sample,
    )

    # Evaluate the model
    model.evaluate_generator(
        generator=generator_from(csv_rows_test),
        val_samples=FLAGS.testing_sample,
    )

    save(model, FLAGS.save_file)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

import pickle
import tensorflow as tf
import numpy as np
import csv
import pytest;
from keras.optimizers import Adam
from keras.layers import Input, Flatten, Dense, Conv2D, Dropout
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from scipy.misc import imread
from sklearn.model_selection import train_test_split

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', 'vgg_traffic_100_bottleneck_features_train.p', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', 'vgg_traffic_bottleneck_features_validation.p', "Bottleneck features validation file (.p)")
flags.DEFINE_string('weights_file', 'output_model.h5', "The weights file to save (.h5)")
flags.DEFINE_string('model_file', 'output_model.json', "The model file to save")
flags.DEFINE_string('driving_log', 'driving_log.csv', 'The driving log.')
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")
flags.DEFINE_integer('epoch_sample', 10000, 'The epoch sample.')
flags.DEFINE_integer('validation_sample', 10, 'The validation sample.')
flags.DEFINE_integer('testing_sample', 10, 'The testing sample.')
flags.DEFINE_float('lrate', 0.1, 'The learning rate')


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val

def get_pretrained_model():
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    nb_classes = len(np.unique(y_train))

    # define model
    input_shape = X_train.shape[1:]
    inp = Input(shape=input_shape)
    x = Flatten()(inp)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(inp, x)
    model.summary(line_length=150)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train model
    model.fit(X_train, y_train, nb_epoch=FLAGS.epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val),
              shuffle=True)
    return model


def get_vgg_model(image):
    input_image = Input(shape=image.shape)
    vgg16 = VGG16(include_top=False, input_tensor=input_image)

    # for layer in vgg16.layers[:-3]:
    #     layer.trainable = False


    x = vgg16.get_layer("block5_conv3").output
    x = Flatten()(x)
    x = Dense(1, activation='linear')(x)

    model = Model(input=vgg16.input, output=x)

    model.summary(line_length=150)
    model.compile(optimizer='adam', loss='mse')
    return model


def get_other_model():
    model = Sequential([
        Conv2D(32, 3, 3, input_shape=(160, 320, 3), border_mode='same', activation='relu'),
        Conv2D(64, 3, 3, border_mode='same', activation='relu'),
        Dropout(0.5),
        Conv2D(128, 3, 3, border_mode='same', activation='relu'),
        Conv2D(256, 3, 3, border_mode='same', activation='relu'),
        Dropout(0.5),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, name='output', activation='tanh'),
    ])
    model.compile(optimizer=Adam(lr=FLAGS.lrate), loss='mse')
    return model


def set_trainable(model):
    for layer in model.layers[:-3]:
        layer.trainable = False


def load_csv(path):
    csv_rows = []
    with open(path, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            csv_rows.append(row)

    csv_rows_main, csv_rows_test = train_test_split(csv_rows, test_size=0.15)
    csv_rows_train, csv_rows_val = train_test_split(csv_rows_main, test_size=0.15)
    return (csv_rows_train, csv_rows_val, csv_rows_test)


def generator_from(csv_rows):
    for rows in csv_rows:
        image = imread(rows[0]).astype(np.float32)
        image = image - np.mean(image)
        image = image.reshape((1,) + image.shape)

        angle = np.array([float(rows[6]) / 360.0])

        yield (image, angle)


def get_image(path):
    with open(path, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            image = imread(rows[0]).astype(np.float32)
            image = image - np.mean(image)
            # csv.close(infile)
            return image


def save_model(model):
    print("Saving model...")
    model.save_weights(FLAGS.weights_file)

    model_as_json = model.to_json()
    with open(FLAGS.model_file, "w") as model_file:
        model_file.write(model_as_json)
    print("Model saved.")


def main(_):
    (csv_rows_train, csv_rows_val, csv_rows_test) = load_csv('./driving_log.csv')

    image = get_image('./driving_log.csv')
    model = get_vgg_model(image)
    # set_trainable(model)
    # model = get_other_model()

    # Train the model
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

    save_model(model)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

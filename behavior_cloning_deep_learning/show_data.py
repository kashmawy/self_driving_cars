import numpy as np
import csv
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt


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


def load_image(path):
    return imread(path).astype(np.float32)


def load_and_preprocess_image(path):
    return preprocess_image(np.array([load_image(path)]))[0]


def show_images(image):
    plt.imshow(image[0,:,:,0])
    plt.show()
    input("Press any key to continue.")


def save_image(image_name, image):
    imsave(image_name, image)


def get_angels_from_csv(path):
    angels = []

    angle_min = 1000.0
    angle_max = -1000.0

    with open(path, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if row[2] != '':
                angels.append((float(row[3]), True))
            else:
                angels.append((float(row[3]), False))

            angle_min = min(angle_min, float(row[3]))
            angle_max = max(angle_max, float(row[3]))

    return angels


def data_augment(angels):
    for i in range(0, len(angels)):
        if angels[i][1]:
            angels.append((angels[i][0] + 0.25, True))
            angels.append((angels[i][0] - 0.25, True))

        angels.append((angels[i][0] * -1.0, True))

    return angels


def show_histogram(angels):
    plt.hist(angels, 20)
    plt.show()


def main():
    IMG1 = 'IMAGES/analog_center_before_preprocessing.jpg'
    image_preprocessed = load_and_preprocess_image(IMG1)
    save_image('IMAGES/analog_center_after_preprocessing.jpg', image_preprocessed)

    IMG2 = 'IMAGES/digital_center.jpg'
    image_2_data = load_image(IMG2)
    save_image('IMAGES/digital_center_flipped.jpg', flip(image_2_data, 0)[0])

    IMG3 = 'IMAGES/analog_center.jpg'
    image_3_data = load_image(IMG3)
    save_image('IMAGES/analog_center_flipped.jpg', flip(image_3_data, 0)[0])

    angels = get_angels_from_csv('solution/driving_log_track_1.csv')
    angels = data_augment(angels)
    angels = [angel[0] for angel in angels]
    show_histogram(angels)


if __name__ == '__main__':
    main()

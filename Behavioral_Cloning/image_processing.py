
import math
import numpy as np
from scipy.ndimage import rotate
from scipy.misc import imresize
from skimage.util import random_noise
from skimage.io import imsave
from skimage.transform import AffineTransform
from skimage.transform import warp
from skimage import exposure
from skimage import color
import matplotlib.pyplot as plt
import cv2


ROTATION_RANGE = 5
TRANSLATION_RANGE = 10
GAMMA_RANGE = 0.8
SHADOW_RANGE = 0.5
NUMBER_OF_SHADOWS = 5
INPUT_IMAGE_WIDTH = 320.0
INPUT_IMAGE_HEIGHT = 160.0
DISCARDED_IMAGE_MARGIN = 20.0
PREPARED_IMAGE_WIDTH = 200.0
PREPARED_IMAGE_HEIGHT = 66.0


def noisy_rotate_image(image, range):
    """
    Randomly rotates image around center by angle within specified range
    :param image: image
    :param range: rotation range in degrees
    :return: rotated image
    """
    angle = np.random.uniform(-range, range)
    return rotate(image, angle, reshape=False)


def noisy_translate_image(image, range):
    """
    Randomly translates image by angle within specified range
    :param image: image
    :param range: 2D translation range
    :return: translated image
    """
    horizontal = np.random.uniform(-range, range)
    vertical = np.random.uniform(-range, range)
    transform = AffineTransform(translation=(horizontal, vertical))
    return warp(image, transform)
    # return image.transform(image.size, Image.AFFINE, (1, 0, horizontal, 0, 1, vertical))
    # return shift(image, [horizontal, vertical], mode="reflect", prefilter=False)


def convert_to_hls(image):
    """
    Converts image from RGB to HLS
    :param image: image in RGB
    :return: image in HLS
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def convert_to_hsv(image):
    """
    Converts image from RGB to HSV
    :param image: image in RGB
    :return: image in HSV
    """
    return color.rgb2hsv(image)


def convert_to_grey(image):
    """
    Converts image from RGB to greyscale
    :param image: image in RGB
    :return: image in greyscale
    """
    return color.rgb2grey(image)


def cast_shadow(image):
    """
    Casts a random rectangular shadow
    :param image: image
    :return: image with a shadow applied
    """
    xa = np.random.uniform(0, INPUT_IMAGE_WIDTH)
    ya = np.random.uniform(0, INPUT_IMAGE_HEIGHT)
    xb = np.random.uniform(0, INPUT_IMAGE_WIDTH)
    yb = np.random.uniform(0, INPUT_IMAGE_HEIGHT)

    x1 = int(min(xa, xb))
    x2 = int(max(xa, xb))
    y1 = int(min(ya, yb))
    y2 = int(max(ya, yb))

    intensity = np.random.uniform(1.0 - SHADOW_RANGE, 1.0)

    for j in range(y1, y2):
        for i in range(x1, x2):
            image[j][i] = image[j][i] * intensity

    return image


def cast_shadows(image):
    number = int(np.random.uniform(0, NUMBER_OF_SHADOWS + 1))
    for i in range(number):
        image = cast_shadow(image)
    return image


def noisy_gamma(image, range):
    """
    Adjust gamma
    :param image: image
    :param range: range for adjusting gamma, [1.0 - range, 1.0 + range]
    :return: image with gamma adjusted
    """
    gamma = np.random.uniform(1.0 - range, 1.0 + range)
    return exposure.adjust_gamma(image, gamma)


def noisy_picture(image):
    """
    Applies random noise to image to simulate bad driving conditions
    :param image: image
    :return: image with noise
    """
    return random_noise(image, mode="s&p")


def resize_image(image, width, height):
    """
    Resizes image
    :param image: image
    :param width: new width
    :param height: new height
    :return: resized image
    """
    return imresize(image, [height, width], interp="lanczos")


def crop_image(image, x, y, width, height):
    """
    Crops image at a given starting position to new width and height
    :param image: image
    :param x: starting x coordinate
    :param y: starting y coordinate
    :param width: new width
    :param height: new height
    :return: cropped image
    """
    return image[y:y + height, x:x + width]


def prepare_image(image):
    """
    Complete preparation pipeline
    :param image: input image
    :return: image with post-processing pipeline applied
    """
    result = image
    result = cast_shadows(image)
    result = noisy_rotate_image(result, ROTATION_RANGE)
    result = noisy_translate_image(result, TRANSLATION_RANGE)
    result = noisy_gamma(result, GAMMA_RANGE)
    width = PREPARED_IMAGE_WIDTH + 2 * DISCARDED_IMAGE_MARGIN
    height = INPUT_IMAGE_HEIGHT * PREPARED_IMAGE_WIDTH / INPUT_IMAGE_WIDTH + 2 * DISCARDED_IMAGE_MARGIN
    result = resize_image(result, int(width), int(height))
    result = crop_image(result, int(DISCARDED_IMAGE_MARGIN),
                        int(height - PREPARED_IMAGE_HEIGHT - DISCARDED_IMAGE_MARGIN),
                        int(PREPARED_IMAGE_WIDTH), int(PREPARED_IMAGE_HEIGHT))
    # result = noisy_picture(result)
    # result = convert_to_hls(result)
    # result = convert_to_hsv(result)

    return result


def replay_image(image):
    """
    Applies image processing to real-time driving images. Can be different to prepare_image
    :param image: real-time driving image
    :return: post-processed image
    """
    result = image
    result = noisy_rotate_image(result, ROTATION_RANGE)
    result = noisy_translate_image(result, TRANSLATION_RANGE)
    # result = noisy_gamma(result, GAMMA_RANGE)
    width = PREPARED_IMAGE_WIDTH + 2 * DISCARDED_IMAGE_MARGIN
    height = INPUT_IMAGE_HEIGHT * PREPARED_IMAGE_WIDTH / INPUT_IMAGE_WIDTH + 2 * DISCARDED_IMAGE_MARGIN
    result = resize_image(result, int(width), int(height))
    result = crop_image(result, int(DISCARDED_IMAGE_MARGIN),
                        int(height - PREPARED_IMAGE_HEIGHT - DISCARDED_IMAGE_MARGIN),
                        int(PREPARED_IMAGE_WIDTH), int(PREPARED_IMAGE_HEIGHT))
    # result = noisy_picture(result)
    return result


def save_image(file_name, image):
    """
    Saves image to a file
    :param file_name: image file name
    :param image: image
    :return: None
    """
    imsave(file_name, image)


#input_file_name = "data/IMG/center_2016_12_01_13_30_48_287.jpg"
# image = cv2.imread(input_file_name)
# image = mpimg.imread(input_file_name)
#image = plt.imread(input_file_name)
# image = np.column_stack((image))
# image = np.dstack((image))
#prepared_image = prepare_image(image)
#save_image("test.png", prepared_image)
# print(image)

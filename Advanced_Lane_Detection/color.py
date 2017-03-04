import cv2
import numpy as np


def hls_select(image, threshold=(0, 255)):
    """
    In HLS model masks pixels with saturation S-value inside threshold range
    :param image: image
    :param threshold: saturation threshold
    :return: mask with pixels with saturation within threshold
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]
    mask = np.zeros_like(s)
    mask[(s > threshold[0]) & (s <= threshold[1])] = 1

    return mask


def pipeline(image, saturation_thresh=(170, 255), sobel_x_thresh=(20, 100)):
    """
    Example image processing pipeline
    TODO: remove
    :param image: image
    :param saturation_thresh: saturation chanel thresholds
    :param sobel_x_thresh: Sobel x threshold
    :return: processed image
    """
    image = np.copy(image)
    # Convert to HLS color space and separate the S channel
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float)
    lightness_channel = hls_image[:, :, 1]
    saturation_channel = hls_image[:, :, 2]
    # Sobel x
    sobel_x = cv2.Sobel(lightness_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    absolute_sobel_x = np.absolute(sobel_x)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel_x = np.uint8(255 * absolute_sobel_x / np.max(absolute_sobel_x))

    # Threshold x gradient
    sobel_x_mask = np.zeros_like(scaled_sobel_x)
    sobel_x_mask[(scaled_sobel_x >= sobel_x_thresh[0]) & (scaled_sobel_x <= sobel_x_thresh[1])] = 1

    # Threshold color channel
    saturation_mask = np.zeros_like(saturation_channel)
    saturation_mask[(saturation_channel >= saturation_thresh[0]) & (saturation_channel <= saturation_thresh[1])] = 1
    # Stack each channel
    # Note color_mask[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_mask = np.dstack((np.zeros_like(sobel_x_mask), sobel_x_mask, saturation_mask))
    return color_mask

import matplotlib.image as mpimg
import cv2
import numpy as np
import matplotlib.pyplot as plt
import color


def load_image(file_name):
    """
    Loads a single image in RGB format
    :param file_name: file name
    :return: image
    """
    return mpimg.imread(file_name)


def load_images(file_names):
    """
    Loads multiple images in RGB format
    :param file_names: array of file names
    :return: array of images
    """
    result = []
    for file_name in file_names:
        result.append(load_image(file_name))
    return result


def absolute_sobel_threshold(image, orientation='x', threshold=(0, 255)):
    """
    Calculates absolute Sobel threshold mask
    :param image: image
    :param orientation: Sobel operator orientation, could be 'x' or 'y'
    :param threshold: threshold range of accepted values
    :return: mask covering pixels within threshold
    """
    gray = image
    if len(gray.shape) == 3 and gray.shape[2] != 1:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if orientation == 'x':
        dx = 1;
        dy = 0;
    else:
        dx = 0;
        dy = 1;
    sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy)
    sobel = np.absolute(sobel)
    sobel = np.uint8(255 * sobel / np.max(sobel))
    mask = np.zeros_like(sobel)
    mask[(sobel >= threshold[0]) & (sobel <= threshold[1])] = 1
    # plt.imshow(mask, cmap='gray')
    return mask


def magnitude_threshold(image, sobel_kernel=3, magnitude_thresh=(0, 255)):
    """
    Calculates Sobel magnitude threshold mask
    :param image: image
    :param sobel_kernel: size of Sobel kernel, default is 3
    :param magnitude_thresh: threshold range of accepted magnitude values
    :return: mask covering pixels within threshold
    """
    gray = image
    if len(gray.shape) == 3 and gray.shape[2] != 1:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    scale = np.max(magnitude / 255)
    magnitude = (magnitude / scale).astype(np.uint8)
    mask = np.zeros_like(magnitude)
    mask[(magnitude >= magnitude_thresh[0]) & (magnitude <= magnitude_thresh[1])] = 1
    return mask


def directional_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """
    Calculates Sobel direction threshold mask
    :param image: image
    :param sobel_kernel: size of Sobel kernel, default is 3
    :param thresh: threshold range of accepted direction values in radians
    :return: mask covering pixels within threshold
    """
    gray = image
    if len(gray.shape) == 3 and gray.shape[2] != 1:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel_x = np.abs(sobel_x)
    abs_sobel_y = np.abs(sobel_y)
    direction = np.arctan2(abs_sobel_y, abs_sobel_x)
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return binary_output


def combine_thresholds(image, kernel_size):
    """
    Combines multiple thresholds together
    :param image: image
    :param kernel_size: Sobel kernel size, default is 3
    :return: mask covering pixels within threshold
    """
    gradient_x_mask = absolute_sobel_threshold(image, orientation='x', threshold=(50, 255))
    gradient_y_mask = absolute_sobel_threshold(image, orientation='y', threshold=(50, 255))
    magnitude_mask = magnitude_threshold(image, sobel_kernel=kernel_size, magnitude_thresh=(60, 255))
    # directional_mask = directional_threshold(image, sobel_kernel=kernel_size, thresh=(0.7, 1.3))
    directional_mask = directional_threshold(image, sobel_kernel=17, thresh=(0.7, 1.3))
    combined_mask = np.zeros_like(directional_mask)
    # combined_mask[((gradient_x_mask == 1) & (gradient_y_mask == 1)) | ((magnitude_mask == 1) & (directional_mask == 1))] = 1
    combined_mask[(gradient_x_mask == 1) | ((magnitude_mask == 1) & (directional_mask == 1))] = 1
    # combined_mask[(gradient_y_mask == 1) | ((magnitude_mask == 1) & (directional_mask == 1))] = 1
    return combined_mask


def channel_threshold(channel, threshold=(0, 255)):
    """
    Calculates mask for a single channel
    :param channel: channel
    :param threshold: threshold range of accepted values
    :return: mask covering pixels within threshold
    """
    mask = np.zeros_like(channel)
    mask[(channel > threshold[0]) & (channel <= threshold[1])] = 1
    return mask


def combine_masks(mask1, mask2, value=1):
    result = np.zeros_like(mask1)
    result[(mask1 != 0) & (mask2 != 0)] = value
    return result


def filtering_pipeline(image, saturation_threshold=(120, 255), lightness_threshold=(30, 255)):
    """
    Complete image filtering pipeline
    :param image: image
    :param saturation_threshold: HLS saturation S-channel threshold
    :param lightness_threshold: HLS lightness L-channel threshold
    :return: filtered image
    """
    image = np.copy(image)
    # Convert to HSV color space and separate the V channel
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float)
    lightness_channel = hls_image[:, :, 1]
    saturation_channel = hls_image[:, :, 2]

    gradient_mask = combine_thresholds(gray_image, 3)

    saturation_mask = channel_threshold(saturation_channel, (saturation_threshold[0], saturation_threshold[1]))
    lightness_mask = channel_threshold(lightness_channel, (lightness_threshold[0], lightness_threshold[1]))
    color_mask = combine_masks(saturation_mask, lightness_mask)

    mask = np.dstack((np.zeros_like(gradient_mask), gradient_mask, color_mask))

    # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    # return color_binary

    return mask


def get_mask(image, maximum=255):
    """
    Gets mask out of non-zero pixels in an image
    :param image: image
    :param maximum: mask value
    :return: mask covering non-zero pixels
    """
    result = np.zeros((image.shape[0], image.shape[1]))
    for row in range(0, image.shape[0]):
        for column in range(0, image.shape[1]):
            if image[row, column, 1] != 0 or image[row, column, 2] != 0:
                result[row, column] = maximum
    return result


def draw_polynomial(image, polynomial):
    """
    Draws a polynomial
    TODO: doesn't work
    :param image: image
    :param polynomial: polynomial
    :return: image
    """
    # Generate x and y values for plotting
    plot_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    plot_x = polynomial[0] * plot_y ** 2 + polynomial[1] * plot_y + polynomial[2]

    plt.imshow(image)
    plt.plot(plot_x, plot_y, color='yellow', linewidth=5.0)
    plt.xlim(0, image.shape[1])
    plt.ylim(0, image.shape[0])
    return image


def draw_lane_overlay(image, transform, left_polynomial, right_polynomial, curve_radius=None, center_offset=None):
    """
    Draws a lane overlay on a straightened image
    :param image: image
    :param transform: transform from lane space to world space
    :param left_polynomial: left lane polynomial
    :param right_polynomial: right lane polynomial
    :param curve_radius: average curve radius
    :param center_offset: car position's center offset
    :return: image with lane information overlayed
    """
    plot_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_plot_x = left_polynomial[0] * plot_y ** 2 + left_polynomial[1] * plot_y + left_polynomial[2]
    right_plot_x = right_polynomial[0] * plot_y ** 2 + right_polynomial[1] * plot_y + right_polynomial[2]

    # Create an image to draw the lines on
    lane_image = np.zeros_like(image).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    points_left = np.array([np.transpose(np.vstack([left_plot_x, plot_y]))])
    points_right = np.array([np.flipud(np.transpose(np.vstack([right_plot_x, plot_y])))])
    points = np.hstack((points_left, points_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(lane_image, np.int_([points]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    projected_lane_image = cv2.warpPerspective(lane_image, transform, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, projected_lane_image, 0.3, 0)

    # add curvature and center offset information
    if (curve_radius is not None) and (center_offset is not None):
        result = cv2.putText(result, 'turn radius=%.2fm  center offset=%.2fm' % (curve_radius, center_offset),
                             (360, 30), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return result

import numpy as np
import matplotlib.pyplot as plt
import cv2


def window_mask(width, height, image, center, level):
    """
    Returns mask for a given window
    :param width: width of the window
    :param height: height of the window
    :param image: image
    :param center: horizontal window center
    :param level: window level on y coordinate (stacked windows on top of each other)
    :return: window mask
    """
    output = np.zeros_like(image)
    output[int(image.shape[0] - (level + 1) * height):int(image.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), image.shape[1])] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin):
    """
    Finds centroids corresponding to a window using convolution
    :param image: image
    :param window_width: width of the window
    :param window_height: height of the window
    :param margin: window margin
    :return:
    """
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window,
        # not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids


def sliding_window_image(image, window_width=50, window_height=80, margin=100):
    """
    Draws sliding windows into an image
    :param image: image
    :param window_width: width of the window
    :param window_height: height of the window
    :param margin: window margin
    :return: image with sliding windows, left points, right points
    """
    # Break image into 9 vertical layers since image height is 720
    # How much to slide left and right for searching
    window_centroids = find_window_centroids(image, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        left_points = np.zeros_like(image)
        right_points = np.zeros_like(image)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            left_mask = window_mask(window_width, window_height, image, window_centroids[level][0], level)
            right_mask = window_mask(window_width, window_height, image, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            left_points[(left_points == 255) | (left_mask == 1)] = 255
            right_points[(right_points == 255) | (right_mask == 1)] = 255

        # Draw the results
        template = np.array(right_points + left_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.array(cv2.merge((image, image, image)),
                           np.uint8)  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the original road image with window results

    # If no window centers found, just display original road image
    else:
        output = np.array(cv2.merge((image, image, image)), np.uint8)
    return output, left_points, right_points


def extract_point_coordinates(image):
    """
    Extracts non-zero pixel coordinates from image
    :param image: image
    :return: non-zero pixel coordinates
    """
    result_x = []
    result_y = []
    for row in range(0, image.shape[0]):
        for column in range(0, image.shape[1]):
            if image[row, column] != 0:
                result_x.append(column)
                result_y.append(row)
    # result_y = result_y[::-1]
    return result_x, result_y


def fit_quadratic_polynomial(points_x, points_y):
    """
    Fits a quadratic polynomial over x and y coordinates with y as the main axis
    :param points_x: x coordinates
    :param points_y: y coordinates
    :return: quadratic coefficients
    """
    return np.polyfit(points_y, points_x, 2)


def search_for_lanes(mask):
    """
    Searches for lanes in a mask from scratch
    :param mask: mask with points that can correspond to lanes
    :return: left curve, right curve, left x coordinates, left y coordinates, right x coordinates, right y coordinates
    """
    # Assuming you have created a warped binary image called "mask"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(mask[mask.shape[0] / 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    output = np.dstack((mask, mask, mask)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    mid_point = np.int(histogram.shape[0]/2)
    left_x_base = np.argmax(histogram[:mid_point])
    right_x_base = np.argmax(histogram[mid_point:]) + mid_point

    # Choose the number of sliding windows
    number_of_windows = 9
    # Set height of windows
    window_height = np.int(mask.shape[0] / number_of_windows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero_pixels = mask.nonzero()
    nonzero_y = np.array(nonzero_pixels[0])
    nonzero_x = np.array(nonzero_pixels[1])
    # Current positions to be updated for each window
    left_x_current = left_x_base
    right_x_current = right_x_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minimum_pixels = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_indices = []
    right_lane_indices = []

    # Step through the windows one by one
    for window in range(number_of_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = mask.shape[0] - (window + 1) * window_height
        win_y_high = mask.shape[0] - window * window_height
        win_x_left_low = left_x_current - margin
        win_x_left_high = left_x_current + margin
        win_x_right_low = right_x_current - margin
        win_x_right_high = right_x_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(output,(win_x_left_low,win_y_low),(win_x_left_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(output,(win_x_right_low,win_y_low),(win_x_right_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_indices = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_x_left_low) & (nonzero_x < win_x_left_high)).nonzero()[0]
        good_right_indices = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_x_right_low) & (nonzero_x < win_x_right_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)
        # If you found > minimum_pixels pixels, recenter next window on their mean position
        if len(good_left_indices) > minimum_pixels:
            left_x_current = np.int(np.mean(nonzero_x[good_left_indices]))
        if len(good_right_indices) > minimum_pixels:
            right_x_current = np.int(np.mean(nonzero_x[good_right_indices]))

    # Concatenate the arrays of indices
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    # Extract left and right line pixel positions
    x_left = nonzero_x[left_lane_indices]
    y_left = nonzero_y[left_lane_indices]
    x_right = nonzero_x[right_lane_indices]
    y_right = nonzero_y[right_lane_indices]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(y_left, x_left, 2)
    right_fit = np.polyfit(y_right, x_right, 2)
    return left_fit, right_fit, x_left, y_left, x_right, y_right


def adjust_lanes(mask, left_curve, right_curve):
    """
    Searches for lanes in a mask with previously known lanes
    :param mask: mask
    :param left_curve: left lane curve
    :param right_curve: right lane curve
    :return: left curve, right curve, left x coordinates, left y coordinates, right x coordinates, right y coordinates
    """
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "mask")
    # It's now much easier to find line pixels!
    non_zero_pixels = mask.nonzero()

    y_non_zero = np.array(non_zero_pixels[0])
    x_non_zero = np.array(non_zero_pixels[1])
    margin = 100
    left_lane_indices = ((x_non_zero > (left_curve[0] * (y_non_zero ** 2) + left_curve[1] * y_non_zero + left_curve[2] - margin)) &
                         (x_non_zero < (left_curve[0] * (y_non_zero ** 2) + left_curve[1] * y_non_zero + left_curve[2] + margin)))
    right_lane_indices = ((x_non_zero > (right_curve[0] * (y_non_zero ** 2) + right_curve[1] * y_non_zero + right_curve[2] - margin)) &
                          (x_non_zero < (right_curve[0] * (y_non_zero ** 2) + right_curve[1] * y_non_zero + right_curve[2] + margin)))

    # Again, extract left and right line pixel positions
    x_left = x_non_zero[left_lane_indices]
    y_left = y_non_zero[left_lane_indices]
    x_right = x_non_zero[right_lane_indices]
    y_right = y_non_zero[right_lane_indices]

    if y_left.shape[0] < 12 or y_right.shape[0] < 12:
        return None, None

    # Fit a second order polynomial to each
    left_curve = np.polyfit(y_left, x_left, 2)
    right_curve = np.polyfit(y_right, x_right, 2)

    return left_curve, right_curve, x_left, y_left, x_right, y_right


def find_lanes(mask, left_curve=None, right_curve=None):
    """
    Finds lanes in a mask
    :param mask: mask
    :param left_curve: left lane polynomial
    :param right_curve: right lane polynomial
    :return: left curve, right curve, left x coordinates, left y coordinates, right x coordinates, right y coordinates
    """
    if (left_curve is not None) and (right_curve is not None):
        # try to reuse previous lanes
        left_curve, right_curve, x_left, y_left, x_right, y_right = adjust_lanes(mask, left_curve, right_curve)
        if (left_curve is not None) and (right_curve is not None):
            return left_curve, right_curve, x_left, y_left, x_right, y_right
    # search for lanes anew
    left_curve, right_curve, x_left, y_left, x_right, y_right = search_for_lanes(mask)
    return left_curve, right_curve, x_left, y_left, x_right, y_right

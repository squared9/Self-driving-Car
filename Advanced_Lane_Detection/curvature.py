import numpy as np


radius_weight = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
radius_sum = [10.0, 19.0, 27.0, 34.0, 40.0, 45.0, 49.0, 52.0, 54.0, 55.0]
radius_values = []


def compute_pixel_space_curve_radius(left_fit, right_fit, dimension_y):
    """
    Computes curve radius in pixel space
    :param left_fit: left polynomial
    :param right_fit: right_polynomial
    :param dimension_y: vertical dimension
    :return: average curve radius, left curve radius and right curve radius
    """
    y_eval = dimension_y
    left_curve_radius = ((1 + (2 * left_fit[0] * y_eval + left_fit[1])**2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curve_radius = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    average_curve_radius = (left_curve_radius + right_curve_radius) / 2
    return average_curve_radius, left_curve_radius, right_curve_radius


def compute_curve_radius(x_left, y_left, x_right, y_right, dimension_y):
    """
    Computes curve radius in world space
    :param x_left: x coordinates of left points
    :param y_left: y coordinates of left points
    :param x_right: x coordinates of right points
    :param y_right: y coordinates of right points
    :param dimension_y: vertical dimension
    :return: average curve radius, left curve radius and right curve radius
    """
    # Define conversions in x and y from pixels space to meters
    y_meters_per_pixel = 30 / 720  # meters per pixel in y dimension
    x_meters_per_pixel = 3.7 / 700  # meters per pixel in x dimension

    y_eval = dimension_y

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(y_left * y_meters_per_pixel, x_left * x_meters_per_pixel, 2)
    right_fit_cr = np.polyfit(y_right * y_meters_per_pixel, x_right * x_meters_per_pixel, 2)

    # Calculate the new radii of curvature
    left_curve_radius = ((1 + (2 * left_fit_cr[0] * y_eval * y_meters_per_pixel + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curve_radius = ((1 + (2 * right_fit_cr[0] * y_eval * y_meters_per_pixel + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    average_curve_radius = (left_curve_radius + right_curve_radius) / 2

    average_curve_radius = average_radius(average_curve_radius)

    return average_curve_radius, left_curve_radius, right_curve_radius


def compute_center_offset(width, height, left_fit, right_fit):
    """
    Computes car's offset from center of the lane
    :param width:
    :param height:
    :param left_fit:
    :param right_fit:
    :return:
    """
    x_meters_per_pixel = 3.7 / 700  # meters per pixel in x dimension

    y_max = height - 1
    x_left = left_fit[0] * (y_max ** 2) + left_fit[1] * y_max + left_fit[2]
    x_right = right_fit[0] * (y_max ** 2) + right_fit[1] * y_max + right_fit[2]
    result = width / 2 - (x_left + x_right)/2
    result *= x_meters_per_pixel

    return result


def average_radius(radius):
    """
    Averages radius over last 10 values using linearly decreasing weights from 10 (most recent) to 1 (oldest)
    :param radius: radius
    :return: averaged radius
    """
    global radius_values
    global radius_sum
    global radius_weight

    radius_values = [radius] + radius_values
    if len(radius_values) > 10:
        del radius_values[-1]

    sum = 0.0
    for i in range(0, len(radius_values)):
        sum += radius_values[i] * radius_weight[i]

    sum /= radius_sum[len(radius_values) - 1]

    return sum

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.image as mpimg


def calibrate_camera(camera_directory, columns, rows):
    """
    Calibrates camera
    :param camera_directory: directory containing only checkerboard images for calibration
    :param columns: number of checkerboard calibration columns
    :param rows: number of checkerboard calibration rows
    :return: inverse distortion projection camera matrix, distortion coefficients
    """
    # Make a list of calibration images
    calibration_image_file_names = [file_name for file_name in listdir(camera_directory) if isfile(join(camera_directory, file_name))]

    object_space_points = []
    image_points = []
    for file_name in calibration_image_file_names:
        # generate equidistant object-space points covering expected grid
        object_points = np.zeros((columns * rows, 3), np.float32)
        object_points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

        image = cv2.imread(join(camera_directory, file_name))

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners2
        success, corners = cv2.findChessboardCorners(gray_image, (columns, rows), None)

        # If found, append corners and object points
        if success:
            image_points.append(corners)
            object_space_points.append(object_points)
    success, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(object_space_points, image_points, gray_image.shape[::-1], None, None)
    if success:
        return camera_matrix, distortion_coefficients
    return None


def straighten_image_projection(image, camera_matrix, distortion_coefficients):
    """
    Straightens image projection
    :param image: image
    :param camera_matrix: inverse distortion projection camera matrix
    :param distortion_coefficients: distortion coefficients
    :return: straightened image
    """
    result = cv2.undistort(image, camera_matrix, distortion_coefficients)
    return result


def get_perspective(source_corners, destination_corners):
    """
    Returns perspective projection transform computed from before and after points
    :param source_corners: before points
    :param destination_corners: after points
    :return: perspective projection transform
    """
    perspective_transform = cv2.getPerspectiveTransform(source_corners, destination_corners)
    return perspective_transform


def warp_image(image, perspective_transform):
    """
    Warps image according to perspective projection transform
    :param image: image
    :param perspective_transform: perspective projection transform
    :return: warped image according to perspective projection transform
    """
    warped_image = cv2.warpPerspective(image, perspective_transform, (image.shape[1], image.shape[0]))
    return warped_image


def warp(image, source_corners, destination_corners, camera_matrix, distortion_coefficients):
    """
    Warps image according to before/after points and camera characteristics
    :param image: image
    :param source_corners: before points
    :param destination_corners: after points
    :param camera_matrix: inverse distortion camera matrix
    :param distortion_coefficients: distortion coefficients
    :return: warped image
    """
    straightened_image = straighten_image_projection(image, camera_matrix, distortion_coefficients)
    perspective_transform = get_perspective(source_corners, destination_corners)
    warped_image = cv2.warpPerspective(straightened_image, perspective_transform,
                                       (straightened_image.shape[1], straightened_image.shape[0]))
    return warped_image, perspective_transform


def calibrate_and_warp(image, object_space_points, image_points):
    """
    Warps image after performing camera calibration according to before/after points
    :param image: image
    :param object_space_points: before points
    :param image_points: after points
    :return: warped image
    """
    # Use cv2.calibrateCamera and cv2.undistort()
    result, matrix, distortion_coefficients, rotation_vectors, translation_vectors = \
        cv2.calibrateCamera(object_space_points, image_points, image.shape[:2], None, None)
    height,  width = image.shape[:2]
    # new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion_coefficients, (width, height), 1,
    #                                                      (width, height))
    # undist = cv2.undistort(img, matrix, distortion_coefficients, None, new_camera_matrix)
    undist = cv2.undistort(image, matrix, distortion_coefficients)
    return undist



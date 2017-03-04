from moviepy.editor import VideoFileClip
import matplotlib.image
import numpy as np
import image_processing
import camera
import constants
import color
import lane_detection
import curvature
import cv2
import matplotlib.image as mpimg

camera_matrix = None
distortion_coefficients = None
# source_corners = np.float32([[120, 720], [1220, 720], [575, 450], [690, 450]])
source_corners = np.float32([[60, 720], [1210, 720], [590, 450], [690, 450]])
# source_corners = np.float32([[60, 720], [1220, 720], [590, 450], [690, 450]])
destination_corners = np.float32([[280, 720], [1000, 720], [280, 0], [1000, 0]])
perspective_transform = None
inverse_perspective_transform = None
left_curve = None
right_curve = None
frame = 0

def initialize_pipeline():
    '''
    Initializes image processing pipeline
    :return: None
    '''
    # calibrate camera first
    global camera_matrix
    global distortion_coefficients
    camera_matrix, distortion_coefficients = camera.calibrate_camera(constants.CALIBRATION_IMAGE_DIRECTORY,
                                                                     constants.CALIBRATION_CHECKERBOARD_COLUMNS,
                                                                     constants.CALIBRATION_CHECKERBOARD_ROWS)
    global perspective_transform
    perspective_transform = camera.get_perspective(source_corners, destination_corners)

    global inverse_perspective_transform
    inverse_perspective_transform = camera.get_perspective(destination_corners, source_corners)

    return None


def process_frame(image):
    '''
    Image processing pipeline
    :param image: input frame
    :return: processed frame
    '''
    global frame
    do_save = False
    # do_save = True
    # do_save = frame == 575
    # do_save = frame == 42 * 25 - 2
    # do_save = frame == 18 * 25

    width = image.shape[1]
    height = image.shape[0]

    use_convolution = False
    # use_convolution = True

    if do_save:
        matplotlib.image.imsave("01-original.png", image)
    straightened = camera.straighten_image_projection(image, camera_matrix, distortion_coefficients)
    if do_save:
        matplotlib.image.imsave("02-straight.png", straightened)
    image_processing_threshold = image_processing.combine_thresholds(straightened, 3)
    if do_save:
        matplotlib.image.imsave("03-image-processing-threshold.png", image_processing_threshold)
    # color_threshold = color.pipeline(straightened)
    # if do_save:
    #     matplotlib.image.imsave("03-color-threshold.png", color_threshold)
    combined_mask = image_processing.filtering_pipeline(straightened)
    if do_save:
        matplotlib.image.imsave("04-combined-mask.png", combined_mask)
    warped = camera.warp_image(combined_mask, perspective_transform)
    if do_save:
        matplotlib.image.imsave("05-warped.png", warped)
    warped_mask = image_processing.get_mask(warped)
    if do_save:
        matplotlib.image.imsave("06-warped_mask.png", np.array(cv2.merge((warped_mask, warped_mask, warped_mask)), np.uint8))

    global left_curve
    global right_curve
    x_left = None
    y_left = None
    x_right = None
    y_right = None

    if not use_convolution:
        left_curve, right_curve, x_left, y_left, x_right, y_right = lane_detection.find_lanes(warped_mask, left_curve, right_curve)
    else:
        lane_image, left_points, right_points = lane_detection.sliding_window_image(warped_mask)
        x_left, y_left = lane_detection.extract_point_coordinates(left_points)
        x_right, y_right = lane_detection.extract_point_coordinates(right_points)
        left_curve = lane_detection.fit_quadratic_polynomial(x_left, y_left)
        right_curve = lane_detection.fit_quadratic_polynomial(x_right, y_right)

    curve_radius = None
    if x_left is not None and x_right is not None:
        average_curve_radius, left_curve_radius, right_curve_radius = curvature.compute_curve_radius(x_left, y_left,
                                                                                                     x_right, y_right,
                                                                                                     height)

    center_offset = None
    if left_curve is not None and right_curve is not None:
        center_offset = curvature.compute_center_offset(width, height, left_curve, right_curve)

    straightened_lane_image = image_processing.draw_lane_overlay(straightened, inverse_perspective_transform,
                                                                 left_curve, right_curve, average_curve_radius,
                                                                 center_offset)
    if do_save:
        matplotlib.image.imsave("08-straightened-lane.png", straightened_lane_image)

    result = straightened_lane_image

    if do_save:
        exit()  # for testing quit after a single frame

    frame += 1

    return result


def process_video(input_file_name, output_file_name):
    '''
    Processes video frame by frame
    :param input_file_name: input video filename
    :param output_file_name: output video filename
    '''
    initialize_pipeline()
    video = VideoFileClip(input_file_name)
    result = video.fl_image(process_frame)
    result.write_videofile(output_file_name, audio=False)


process_video("project_video.mp4", "lane_output.mp4")

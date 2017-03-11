from moviepy.editor import VideoFileClip
import matplotlib.image
import classification
import tracking


frame = 0
classifier = None
scaler = None


def initialize():
    """
    Initializes pipeline by training a classifier and preparing a scaler
    :return: None
    """
    global classifier
    global scaler
    classifier, scaler = classification.train_classifier()
    tracking.windows, tracking.hierarchy_of_windows = tracking.generate_sliding_windows(1280, 720)


def process_frame(image):
    '''
    Image processing pipeline
    :param image: input frame
    :return: processed frame
    '''
    global frame
    global classifier
    global scaler
    do_save = False
    # do_save = True
    # do_save = frame == 575
    # do_save = frame == 42 * 25 - 2
    # do_save = frame == 18 * 25
    # do_save = frame == 12 * 25

    # draw all sliding windows
    if do_save:
        boxes = tracking.draw_hierarchy_of_boxes(image, tracking.hierarchy_of_windows)
        matplotlib.image.imsave("01-boxes.png", boxes)

    # vehicle_windows = tracking.search_windows(image, tracking.windows, classifier, scaler)
    # image = tracking.draw_boxes(image, vehicle_windows)
    # image = tracking.find_all_cars(image, classifier, scaler)

    vehicle_windows = tracking.find_all_vehicles(image, classifier, scaler)
    # image = tracking.draw_boxes(image, vehicle_windows, (255, 200, 200), 1)

    if do_save:
        matplotlib.image.imsave("02-detections.png", image)

    detections = tracking.combine_detections(image, vehicle_windows)
    image = tracking.draw_boxes(image, detections)

    if do_save:
        matplotlib.image.imsave("03-vehicles.png", image)

    if do_save:
        exit()

    frame += 1

    result = image

    return result


def process_video(input_file_name, output_file_name):
    '''
    Processes video frame by frame
    :param input_file_name: input video filename
    :param output_file_name: output video filename
    '''
    initialize()
    print("Processing video...")
    video = VideoFileClip(input_file_name)
    result = video.fl_image(process_frame)
    result.write_videofile(output_file_name, audio=False)


process_video("project_video.mp4", "vehicle_tracking.mp4")
# process_video("test_video.mp4", "vehicle_tracking.mp4")
# process_video("problem_1.mp4", "vehicle_tracking.mp4")

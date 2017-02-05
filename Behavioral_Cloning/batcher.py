import matplotlib.pyplot as plt
import numpy as np
import image_processing


SIDE_CAMERA_ANGLE_ADJUSTMENT = 0.3


def next(driving_log, batch_amount=128, image_data_path="data/"):
    """
    Generator's next method
    :param batch_amount: number of elements per batch
    :param data_path: path to data files
    :return: batch of images and angles
    """
    while True:
        images = []
        angles = []
        image_data = get_image_data(driving_log, batch_amount)
        for file_name, angle in image_data:
            image = plt.imread(image_data_path + file_name)
            adjusted_image = image_processing.prepare_image(image)
            images.append(adjusted_image)
            angles.append(angle)
        yield np.array(images), np.array(angles)


def get_image_data(driving_log, batch_amount=128):
    """
    retrieves image file names and angles
    :param driving_log: driving log containing angles and image names
    :param batch_amount: number of elements per batch
    :return: batch of file names and angles corresponding to training driving run
    """
    batch_indices = np.random.randint(0, len(driving_log), batch_amount)
    image_data = []
    for index in batch_indices:
        file_name = ""
        angle = 0
        position = np.random.randint(0, 3)
        if position == 0:
            file_name = driving_log.iloc[index]['left'].strip()
            angle = driving_log.iloc[index]['steering'] + SIDE_CAMERA_ANGLE_ADJUSTMENT
        elif position == 1:
            file_name = driving_log.iloc[index]['center'].strip()
            angle = driving_log.iloc[index]['steering']
        else:
            file_name = driving_log.iloc[index]['right'].strip()
            angle = driving_log.iloc[index]['steering'] - SIDE_CAMERA_ANGLE_ADJUSTMENT
        image_data.append((file_name, angle))
    return image_data

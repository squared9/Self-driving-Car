import time
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import constants


def get_imagery_file_names():
    """
    Retrieves imagery file names for both vehicles and non-vehicles
    :return: two arrays with imagery file names, one for vehicles and the other for non-vehicles
    """
    vehicle_file_names = glob.glob(constants.VEHICLE_DIRECTORY + constants.IMAGE_EXTENSION)
    non_vehicle_file_names = glob.glob(constants.NON_VEHICLE_DIRECTORY + constants.IMAGE_EXTENSION)
    vehicles = []
    non_vehicles = []

    for file_name in vehicle_file_names:
        vehicles.append(file_name)

    for file_name in non_vehicle_file_names:
        non_vehicles.append(file_name)

    return vehicles, non_vehicles


def randomize_order(array):
    """
    Shuffles order of items in an array randomly
    :param array: array
    :return: randomly reordered array
    """
    np.random.shuffle(array)


def split(features, values, factor=0.8):
    """
    Splits training features and values in two depending on factor
    :param features: training features
    :param values: training values
    :param factor: factor
    :return: train and test sets
    """
    random_state = np.random.randint(0, 100)
    x_train, x_test, y_train, y_test = train_test_split(features, values, test_size=1-factor, random_state=random_state)
    return x_train, y_train, x_test, y_test


# Define a function to compute binned color features
def get_spatial_binning_features(image, size=(32, 32)):
    """
    Computes spatial binning features by downsizing an image, for use by a classifier
    :param image: image
    :param size: dimensions of downsampled image
    :return: downsampled channels of image
    """
    color1 = cv2.resize(image[:, :, 0], size).ravel()
    color2 = cv2.resize(image[:, :, 1], size).ravel()
    color3 = cv2.resize(image[:, :, 2], size).ravel()
    result = np.hstack((color1, color2, color3))
    return result


def convert_color(image, conversion='RGB2YCrCb'):
    """
    Converts image to a different color model
    :param image: image
    :param conversion: OpenCV color conversion specification
    :return: image in desired color model
    """
    if conversion == 'RGB2YCrCb':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    if conversion == 'BGR2YCrCb':
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    if conversion == 'RGB2LUV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)


def get_color_histogram_features(image, number_of_bins=32, bins_range=(0, 256)):
    """
    Computes color histogram features
    :param image: image
    :param number_of_bins: number of histogram bins
    :param bins_range: bin value range
    :return: red, green, blue histogram, bin centers and histogram features
    """
    # Compute the histogram of the RGB channels separately
    red_histogram = np.histogram(image[:, :, 0], bins=number_of_bins, range=bins_range)
    green_histogram = np.histogram(image[:, :, 1], bins=number_of_bins, range=bins_range)
    blue_histogram = np.histogram(image[:, :, 2], bins=number_of_bins, range=bins_range)
    # Generating bin centers
    bin_edges = red_histogram[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    # Concatenate the histograms into a single feature vector
    histogram_features = np.concatenate((red_histogram[0], green_histogram[0], blue_histogram[0]))
    # Return the individual histograms, bin_centers and feature vector
    return red_histogram, green_histogram, blue_histogram, bin_centers, histogram_features


def get_hog_features(image, number_of_orientations, pixels_per_cell, cells_per_block, visualize=False,
                     feature_vector=True):
    """
    Computes Histogram of Oriented Gradients (HOG) features
    :param image: image
    :param number_of_orientations: number of gradient orientations
    :param pixels_per_cell: pixels per cell
    :param cells_per_block: cells per block
    :param visualize: flag specifying if HOG should be returned as an image as well
    :param feature_vector: flag specifying if HOG should be flattened
    :return: HOG features and image with HOG if visualize is set to True
    """
    if visualize:
        features, hog_image = hog(image,
                                  orientations=number_of_orientations,
                                  pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                                  cells_per_block=(cells_per_block, cells_per_block),
                                  transform_sqrt=False,
                                  visualise=True,
                                  feature_vector=False)
        return features, hog_image
    else:
        features = hog(image,
                       orientations=number_of_orientations,
                       pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                       cells_per_block=(cells_per_block, cells_per_block),
                       transform_sqrt=False,
                       visualise=False,
                       feature_vector=feature_vector)
        return features


def get_all_hog_features(image, number_of_orientations, pixels_per_cell, cells_per_block):
    """
    Computes all HOG features
    :param image: image
    :param number_of_orientations: number of gradient orientations
    :param pixels_per_cell: pixels per cell
    :param cells_per_block: cells per block
    :return: HOG features
    :return:
    """
    # canvas_to_search = convert_color(image, conversion='RGB2YCrCb')
    canvas_to_search = image
    channel_1 = canvas_to_search[:, :, 0]
    channel_2 = canvas_to_search[:, :, 1]
    channel_3 = canvas_to_search[:, :, 2]
    hog_1 = get_hog_features(channel_1, number_of_orientations, pixels_per_cell, cells_per_block,
                             feature_vector=True)
    hog_2 = get_hog_features(channel_2, number_of_orientations, pixels_per_cell, cells_per_block,
                             feature_vector=True)
    hog_3 = get_hog_features(channel_3, number_of_orientations, pixels_per_cell, cells_per_block,
                             feature_vector=True)

    hog_features = np.hstack((hog_1.ravel(), hog_2.ravel(), hog_3.ravel()))
    return hog_features


def extract_features(image_file_names, color_space='RGB', spatial_size=(32, 32),
                     histogram_bins=32, histogram_range=(0, 256),
                     hog_orientations=9, hog_pixels_per_cell=8, hog_cells_per_block=2):
    """
    Extracts all features from an image (spatial binning, color histogram and HOG)
    :param image_file_names: image file names
    :param color_space: color space
    :param spatial_size: spatial binning downsampling size
    :param histogram_bins: number of histogram bins
    :param histogram_range: range of histogram bin values
    :param hog_orientations: number of HOG orientations
    :param hog_pixels_per_cell: number of HOG pixels per cell
    :param hog_cells_per_block: number of HOG cells per block
    :return: vector of concatenated spatial binning, color histogram and HOG features
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file_name in image_file_names:
        # Read in each one by one
        image = mpimg.imread(file_name)
        if not file_name.lower().endswith("png"):
            image = image.astype(np.float32) / 255
        # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)
        # Apply bin_spatial() to get spatial color features
        spatial_features = get_spatial_binning_features(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        red_histogram, green_histogram, blue_histogram, bin_centers, histogram_features = get_color_histogram_features(
            feature_image, number_of_bins=histogram_bins, bins_range=histogram_range)

        # hog_features = get_hog_features(gray_image, hog_orientations, hog_pixels_per_cell, hog_cells_per_block,
        #                                 visualize=False, feature_vector=True)
        hog_features = get_all_hog_features(feature_image, hog_orientations, hog_pixels_per_cell, hog_cells_per_block)

        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, histogram_features, hog_features)))
    # Return list of feature vectors
    return features


def combine_features(feature_list):
    """
    Combines features by preparing a suitable scaler
    :param feature_list: list of features to combine
    :return: combined scaled features and corresponding scaler
    """
    features = np.vstack(feature_list).astype(np.float64)
    feature_scaler = StandardScaler().fit(features)
    # Apply the scaler to X
    scaled_features = feature_scaler.transform(features)
    return scaled_features, feature_scaler


def train_classifier():
    """
    Trains a SVM classifier
    :return: SVM classifier
    """
    print("Training classifier...")
    # get file names corresponding to vehicles and non-vehicles
    vehicle_file_names, non_vehicle_file_names = get_imagery_file_names()
    # randomize order of file names
    print("  - randomizing")
    randomize_order(vehicle_file_names)
    randomize_order(non_vehicle_file_names)
    # extract features (binning, histogram, HOG)
    print("  - extracting features")
    print("    - vehicles")
    start_time = time.time()
    vehicle_features = extract_features(vehicle_file_names, color_space=constants.FEATURE_COLOR_SPACE)
    end_time = time.time()
    print("      - extracted in", (round(end_time - start_time, 2)), "seconds")
    print("    - non-vehicles")
    start_time = time.time()
    non_vehicle_features = extract_features(non_vehicle_file_names, color_space=constants.FEATURE_COLOR_SPACE)
    end_time = time.time()
    print("      - extracted in", (round(end_time - start_time, 2)), "seconds")
    # prepare values
    vehicle_values = np.ones(len(vehicle_features))
    non_vehicle_values = np.zeros(len(non_vehicle_features))
    # split to train/test features and values
    train_vehicle_features, train_vehicle_values, test_vehicle_features, test_vehicle_values = split(vehicle_features,
                                                                                                     vehicle_values)
    train_non_vehicle_features, train_non_vehicle_values, test_non_vehicle_features, test_non_vehicle_values = split(non_vehicle_features, non_vehicle_values)
    print("  - train set:")
    print("    -", len(train_vehicle_features), "vehicles")
    print("    -", len(train_non_vehicle_features), "non-vehicles")
    # normalize features
    train_features, train_scaler = combine_features([train_vehicle_features, train_non_vehicle_features])
    test_features, test_scaler = combine_features([test_vehicle_features, test_non_vehicle_features])
    # prepare training values
    train_values = np.hstack((train_vehicle_values, train_non_vehicle_values))
    test_values = np.hstack((test_vehicle_values, test_non_vehicle_values))
    # start SVC training
    svc = LinearSVC()
    # Train the SVC
    print("  - training SVC")
    start_time = time.time()
    svc.fit(train_features, train_values)
    end_time = time.time()
    print("    - trained in", (round(end_time - start_time, 2)), "seconds")
    print("  - testing SVC")
    start_time = time.time()
    score = svc.score(test_features, test_values)
    end_time = time.time()
    print("    - test score:", score)
    print("    - tested in", (round(end_time - start_time, 2)), "seconds")
    return svc, train_scaler


def predict(predictor, features):
    """
    Predicts a value using a predictor from features
    :param predictor: predictor
    :param features: features
    :return: predicted category
    """
    return predictor.predict(features)



import numpy as np
import cv2
import classification
import constants
from scipy.ndimage.measurements import label


previous_detections = None
windows = None
hierarchy_of_windows = None
last_heat_maps = []
heat_map_rolling_sum = None


def slide_window(image_width, image_height, x_start_stop=[None, None], y_start_stop=[400, 650],
                 windows_size=(64, 64), window_overlap_factor=(0.5, 0.5)):
    """
    Slides window with a specified size and overlap
    :param image_width: image width
    :param image_height: image height
    :param x_start_stop: x sliding limits
    :param y_start_stop: y sliding limits
    :param windows_size: window size
    :param window_overlap_factor: window overlap factor
    :return: list of windows covering area
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = image_width
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = image_height
    # Compute the span of the region to be searched
    x_span = x_start_stop[1] - x_start_stop[0]
    y_span = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(windows_size[0] * (1 - window_overlap_factor[0]))
    ny_pix_per_step = np.int(windows_size[1] * (1 - window_overlap_factor[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(windows_size[0] * (window_overlap_factor[0]))
    ny_buffer = np.int(windows_size[1] * (window_overlap_factor[1]))
    nx_windows = np.int((x_span - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((y_span - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + windows_size[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + windows_size[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def generate_sliding_windows(image_width, image_height, sizes=constants.SLIDING_WINDOWS_SIZES,
                             x_ranges=constants.SLIDING_WINDOWS_X_RANGES,
                             y_ranges=constants.SLIDING_WINDOWS_Y_RANGES):
    """
    Generates sliding windows in multiple scales and coverage
    :param image_width: image width
    :param image_height: image height
    :param sizes: window sizes
    :param x_ranges: window x ranges
    :param y_ranges: window y ranges
    :return: windows covering areas
    """
    result = []
    for i in range(len(sizes)):
        size = sizes[i]
        shape = (size, size)
        x_range = x_ranges[i]
        y_range = y_ranges[i]
        window_grid = slide_window(image_width, image_height, x_start_stop=x_range,
                                   y_start_stop=y_range, windows_size=shape)
        result.append(window_grid)

    return sum(result, []), result


def get_image_features(image, color_space='RGB', spatial_size=(32, 32),
                       histogram_bins=32, orientations=9,
                       pixels_per_cell=8, cells_per_block=2, hog_channel=0,
                       spatial_feature=True, histogram_feature=True, hog_feature=True):
    """
    Extracts image features
    :param image: image
    :param color_space: color space
    :param spatial_size: spatial binning size
    :param histogram_bins: number of histogram bins
    :param orientations: number of HOG orientations
    :param pixels_per_cell: number of HOG pixels per cell
    :param cells_per_block: number of HOG cells per block
    :param hog_channel: HOG channel (0-2 or 'ALL')
    :param spatial_feature: flag indicating extracting spatial binning features is needed
    :param histogram_feature: flag indicating extracting color histogram features is needed
    :param hog_feature: flag indicating extracting HOG features is needed
    :return: list with concatenated desired image features
    """
    # 1) Define an empty list to receive features
    image_features = []
    # 2) Apply color conversion if other than 'RGB'
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
    # 3) Compute spatial features if flag is set
    if spatial_feature:
        spatial_features = classification.get_spatial_binning_features(feature_image, size=spatial_size)
        # 4) Append features to list
        image_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if histogram_feature:
        red_histogram, green_histogram, blue_histogram, bin_centers, histogram_features = \
            classification.get_color_histogram_features(feature_image, number_of_bins=histogram_bins)
        # 6) Append features to list
        image_features.append(histogram_features)
    # 7) Compute HOG features if flag is set
    if hog_feature:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(
                    classification.get_hog_features(feature_image[:, :, channel], orientations, pixels_per_cell,
                                                    cells_per_block, visualize=False, feature_vector=True))
        else:
            hog_features = classification.get_hog_features(feature_image[:, :, hog_channel], orientations,
                                                           pixels_per_cell, cells_per_block, visualize=False,
                                                           feature_vector=True)
        # 8) Append features to list
        image_features.append(np.ravel(hog_features))

    # 9) Return concatenated array of features
    return np.concatenate(image_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(image, windows, classifier, scaler, color_space=constants.HISTOGRAM_COLOR_SPACE,
                   spatial_size=constants.SPATIAL_BINNING_SIZE, histogram_bins=constants.HISTOGRAM_NUMBER_OF_BINS,
                   histogram_range=constants.HISTOGRAM_COLOR_RANGE, orientations=constants.HOG_NUMBER_OF_ORIENTATIONS,
                   pixels_per_cell=constants.HOG_PIXELS_PER_CELL, cells_per_block=constants.HIG_CELLS_PER_BLOCK,
                   hog_channel=0, spatial_feature=True,
                   histogram_feature=True, hog_feature=True):
    """
    Searches all windows for matches
    :param image: image
    :param windows: windows
    :param classifier: feature classifier
    :param scaler: feature scaler
    :param color_space: color space
    :param spatial_size: spatial binning size
    :param histogram_bins: number of histogram bins
    :param orientations: number of HOG orientations
    :param pixels_per_cell: number of HOG pixels per cell
    :param cells_per_block: number of HOG cells per block
    :param hog_channel: HOG channel (0-2 or 'ALL')
    :param spatial_feature: flag indicating extracting spatial binning features is needed
    :param histogram_feature: flag indicating extracting color histogram features is needed
    :param hog_feature: flag indicating extracting HOG features is needed
    :return: classification matches
    """
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        classification_image = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = get_image_features(classification_image, color_space=color_space, spatial_size=spatial_size,
                                      histogram_bins=histogram_bins, orientations=orientations,
                                      pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                                      hog_channel=hog_channel, spatial_feature=spatial_feature,
                                      histogram_feature=histogram_feature, hog_feature=hog_feature)
        # 5) Scale extracted features to be fed to classifier
        features_for_test = np.array(features).reshape(1, -1)
        # features_for_test = features.reshape(1, -1)
        test_features = scaler.transform(features_for_test)
        # 6) Predict using your classifier
        prediction = classifier.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def draw_boxes(image, bounding_boxes, color=(0, 0, 255), thickness=6):
    """
    Draws bounding boxes into image
    :param image: image
    :param bounding_boxes: list of bounding boxes ((min X, min Y), (max X, max Y))
    :param color: line color
    :param thickness: line thickness
    :return: image with bounding boxes drawn
    """
    # Make a copy of the image
    result = np.copy(image)
    # Iterate through the bounding boxes
    for bounding_box in bounding_boxes:
        # Draw a rectangle given bounding_box coordinates
        cv2.rectangle(result, bounding_box[0], bounding_box[1], color, thickness)
    # Return the image copy with boxes drawn
    return result


def draw_hierarchy_of_boxes(image, hierarchy_of_bounding_boxes, thickness=1):
    """
    Draws a hierarchy of boxes for sliding windows
    :param image: image
    :param hierarchy_of_bounding_boxes: list of lists of bounding boxes for each window size
    :param thickness: line thickness
    :return: image with a hierarchy of boxes drawn
    """
    # Make a copy of the image
    result = np.copy(image)
    # Iterate through the bounding boxes
    for i in range(len(hierarchy_of_bounding_boxes)):
        bounding_boxes = hierarchy_of_bounding_boxes[i]
        color = constants.SLIDING_WINDOWS_COLORS[i]
        for bounding_box in bounding_boxes:
            # Draw a rectangle given bounding_box coordinates
            cv2.rectangle(result, bounding_box[0], bounding_box[1], color, thickness)
    # Return the image copy with boxes drawn
    return result


def find_vehicle_windows(image, y_start, y_stop, x_start, x_stop, scale, classifier, scaler,
                         orientations=constants.HOG_NUMBER_OF_ORIENTATIONS,
                         pixels_per_cell=constants.HOG_PIXELS_PER_CELL,
                         cells_per_block=constants.HIG_CELLS_PER_BLOCK,
                         spatial_size=constants.SPATIAL_BINNING_SIZE,
                         histogram_bins=constants.HISTOGRAM_NUMBER_OF_BINS):
    """
    Finds all vehicle windows
    :param image: image
    :param y_start: minimal value of y for search
    :param y_stop: maximal value of y for search
    :param x_start: minimal value of x for search
    :param x_stop: maximal value of x for search
    :param scale: classification window scale
    :param classifier: feature classifier
    :param scaler: feature scaler
    :param orientations: number of HOG orientations
    :param pixels_per_cell: number of HOG pixels per cell
    :param cells_per_block: number of HOG cells per block
    :param spatial_size: spatial binning size
    :param histogram_bins: number of color histogram bins
    :return: list of bounding boxes corresponding to found vehicles
    """
    result = []
    image = image.astype(np.float32) / 255

    # image_to_search = image[y_start:y_stop, :, :]
    image_to_search = image[y_start: y_stop, x_start: x_stop, :]
    canvas_to_search = classification.convert_color(image_to_search, conversion='RGB2YCrCb')
    if scale != 1:
        image_shape = canvas_to_search.shape
        canvas_to_search = cv2.resize(canvas_to_search, (np.int(image_shape[1] / scale), np.int(image_shape[0] / scale)))

    channel_1 = canvas_to_search[:, :, 0]
    channel_2 = canvas_to_search[:, :, 1]
    channel_3 = canvas_to_search[:, :, 2]

    # Define blocks and steps as above
    x_number_of_blocks = (channel_1.shape[1] // pixels_per_cell) - 1
    y_number_of_blocks = (channel_1.shape[0] // pixels_per_cell) - 1
    number_of_features_per_block = orientations * cells_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = constants.TRACKING_BASE_WINDOW_DIMENSION
    number_of_blocks_per_window = (window // pixels_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    x_number_of_steps = (x_number_of_blocks - number_of_blocks_per_window) // cells_per_step
    y_number_of_steps = (y_number_of_blocks - number_of_blocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog_1 = classification.get_hog_features(channel_1, orientations, pixels_per_cell, cells_per_block,
                                            feature_vector=False)
    hog_2 = classification.get_hog_features(channel_2, orientations, pixels_per_cell, cells_per_block,
                                            feature_vector=False)
    hog_3 = classification.get_hog_features(channel_3, orientations, pixels_per_cell, cells_per_block,
                                            feature_vector=False)

    for x_block in range(x_number_of_steps):
        for y_block in range(y_number_of_steps):
            y = y_block * cells_per_step
            x = x_block * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog_1[y: y + number_of_blocks_per_window, x: x + number_of_blocks_per_window].ravel()
            hog_feat2 = hog_2[y: y + number_of_blocks_per_window, x: x + number_of_blocks_per_window].ravel()
            hog_feat3 = hog_3[y: y + number_of_blocks_per_window, x: x + number_of_blocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            x_left = x * pixels_per_cell
            y_top = y * pixels_per_cell

            # Extract the image patch
            # sub_canvas = cv2.resize(image_to_search[y_top: y_top + window, x_left: x_left + window], (64, 64))
            sub_canvas = cv2.resize(canvas_to_search[y_top: y_top + window, x_left: x_left + window], (64, 64))

            # Get color features
            spatial_features = classification.get_spatial_binning_features(sub_canvas, size=spatial_size)
            red_histogram, green_histogram, blue_histogram, bin_centers, histogram_features = \
                classification.get_color_histogram_features(sub_canvas, number_of_bins=histogram_bins)

            # Scale features and make a prediction
            features_for_prediction = np.hstack((spatial_features, histogram_features, hog_features)).reshape(1, -1)
            prediction_features = scaler.transform(features_for_prediction)
            prediction = classifier.predict(prediction_features)

            if prediction == 1:
                x_box_left = np.int(x_left * scale)
                y_top_draw = np.int(y_top * scale)
                window_draw = np.int(window * scale)
                result.append(((x_box_left + x_start, y_top_draw + y_start),
                              (x_box_left + window_draw + x_start, y_top_draw + window_draw + y_start)))
    return result


def find_all_vehicles(image, classifier, scaler, orientations=constants.HOG_NUMBER_OF_ORIENTATIONS,
              pixels_per_cell=constants.HOG_PIXELS_PER_CELL, cells_per_block=constants.HIG_CELLS_PER_BLOCK,
              spatial_size=constants.SPATIAL_BINNING_SIZE, histogram_bins=constants.HISTOGRAM_NUMBER_OF_BINS):
    """
    Finds all vehicles
    :param image: image
    :param classifier: feature classifier
    :param scaler: feature scaler
    :param orientations: number of HOG orientations
    :param pixels_per_cell: number of HOG pixels per cell
    :param cells_per_block: number of HOG cells per block
    :param spatial_size: spatial binning size
    :param histogram_bins: number of color histogram bins
    :return: list of bounding boxes corresponding to found vehicles
    """
    result = []
    for i in range(len(constants.SLIDING_WINDOWS_SCALES)):
        scale = constants.SLIDING_WINDOWS_SCALES[i]
        y_range = constants.SLIDING_WINDOWS_Y_RANGES[i]
        x_range = constants.SLIDING_WINDOWS_X_RANGES[i]
        # detected_windows = find_vehicle_windows(image, y_range[0], y_range[1], scale, classifier, scaler)
        # detected_windows = find_vehicle_windows(image, 400, 656, scale, classifier, scaler)
        detected_windows = find_vehicle_windows(image, y_range[0], y_range[1], x_range[0], x_range[1], scale, classifier, scaler)
        result += detected_windows
    return result


def add_heat(heat_map, windows):
    """
    Adds heat information into a heat map
    :param heat_map: heat map
    :param windows: list of hot areas
    :return: updated heat map
    """
    # Iterate through list of bounding boxes
    for box in windows:
        # Add += 1 for all pixels inside each bounding box
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heat map
    return heat_map


def apply_heat_threshold(heat_map, threshold):
    """
    Applies heat threshold
    :param heat_map: heat map
    :param threshold: desired minimal value threshold
    :return: heat map with hot pixels with values above threshold
    """
    # Zero out pixels below the threshold
    heat_map[heat_map <= threshold] = 0
    # Return thresholded map
    return heat_map


def get_heat_labels(heat_map):
    """
    Finds all separate components in a heat map
    :param heat_map: heat map
    :return: map containing labels of unique components comprised of continuous pixel neighbors
    """
    labels = label(heat_map)
    return labels


def get_labeled_windows(labels):
    """
    Computes bounding boxes for labeled components
    :param labels: labeled components
    :return: list of bounding boxes for labeled components
    """
    result = []
    # Iterate through all detected cars
    for vehicle_number in range(1, labels[1]+1):
        # Find pixels with each vehicle_number label value
        non_zero = (labels[0] == vehicle_number).nonzero()
        # Identify x and y values of those pixels
        non_zero_y = np.array(non_zero[0])
        non_zero_x = np.array(non_zero[1])
        # Define a bounding box based on min/max x and y
        bounding_box = ((np.min(non_zero_x), np.min(non_zero_y)), (np.max(non_zero_x), np.max(non_zero_y)))
        if bounding_box[1][0] - bounding_box[0][0] >= constants.TRACKING_MINIMAL_WINDOW_WIDTH and \
            bounding_box[1][1] - bounding_box[0][1] >= constants.TRACKING_MINIMAL_WINDOW_HEIGHT:
            result.append(bounding_box)
    # Return the image
    return result


def combine_heat_maps(heat_map):
    """
    Combines heat maps together using rolling sum and average heat
    :param heat_map: heat map
    :return: average heat map
    """
    global last_heat_maps
    global heat_map_rolling_sum
    if not constants.TRACKING_USE_AVERAGED_HEAT_MAP:
        return heat_map
    if len(last_heat_maps) == 0:
        last_heat_maps = [heat_map]
        heat_map_rolling_sum = heat_map
        return heat_map
    else:
        last_heat_maps.append(heat_map)
        if len(last_heat_maps) > constants.TRACKING_HEAT_MAP_LAYERS:
            heat_map_rolling_sum -= last_heat_maps[0]
            del last_heat_maps[0]
        number_of_layers = len(last_heat_maps)
        heat_map_rolling_sum += heat_map
        result = heat_map_rolling_sum / number_of_layers
        return result


def combine_detections(image, detections):
    """
    Combines vehicle detections
    :param image: image
    :param detections: vehicle detections
    :return: list of unique vehicle detections
    """
    heat_map = np.zeros_like(image[:, :, 0]).astype(np.float)
    add_heat(heat_map, detections)
    heat_map = combine_heat_maps(heat_map)
    apply_heat_threshold(heat_map, constants.TRACKING_HEAT_MAP_THRESHOLD)
    labels = get_heat_labels(heat_map)
    result = get_labeled_windows(labels)
    return result

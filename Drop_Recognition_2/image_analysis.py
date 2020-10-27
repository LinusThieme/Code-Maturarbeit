import numpy as np
from scipy.linalg import lstsq
import cv2
from sklearn.cluster import DBSCAN
import statistics


def add_lines_to_image(image, canny_image):
    for x in range(len(canny_image)):
        for y in range(len(canny_image[0])):
            if canny_image[x, y] == 255:
                image[x, y] = 255
    return image


def remove_zero(image):
    for x in range(len(image)):
        for y in range(len(image[0])):
            if image[x, y] == 0:
                image[x, y] = 1
    return image


def local_average(points, n=2):
    local_averages = []
    for i in range(len(points)):
        if (i < len(points) - (n + 1)) and (i > (n - 1)):
            s = 0
            for j in range(-n, n + 1):
                s += points[i + j]
            local_averages.append(s/(2*n + 1))
        else:
            local_averages.append(None)
    return local_averages


def reduce_list_to_y_value(points):
    y_list = []
    for point in points:
        y_list.append(point[1])
    return y_list


def reduce_list_to_x_value(points):
    x_list = []
    for point in points:
        x_list.append(point[0])
    return x_list


def get_linear_polynominal_coeffs(points):
    # Set up x and y vectors
    x = []
    y = []
    for point in points:
        x.append(point[0])
        y.append(point[1])

    x_vector = np.array(x)
    y_vector = np.array(y)

    # Set up Matrix m for regression
    m = x_vector[:, np.newaxis]**[0, 1]

    # Apply method of least sum of squares
    p, res, rnk, s = lstsq(m, y_vector)

    # Return a,b
    return p


def get_dbscan_clusters_of_line(image, line, epsilon=25, min_sampels=25):
    # Create Dataset for DBSCAN
    rain_video_line_zero_points = []
    for i in range(len(image[line])):
        rain_video_line_zero_points.append([i, image[line][i]])

    # DBSCAN
    cluster_analysis = DBSCAN(eps=epsilon, min_samples=min_sampels).fit_predict(rain_video_line_zero_points)

    # Cluster Sorting
    clusters = []
    for i in range(max(cluster_analysis) + 1):
        clusters.append([])
        for j in range(len(rain_video_line_zero_points)):
            if cluster_analysis[j] == i:
                clusters[i].append(rain_video_line_zero_points[j])

    # Return Cluster
    return clusters


def get_linear_polynominal_coeffs_of_clusters(clusters):
    linear_regression_coeffs = []
    for i in range(len(clusters)):
        linear_regression_coeffs.append(get_linear_polynominal_coeffs(clusters[i]))
    return linear_regression_coeffs


def get_cluster_gaps(clusters):
    gaps = []
    for i in range(len(clusters)):
        if i < len(clusters) - 1:
            gaps.append(clusters[i + 1][0][0] - clusters[i][len(clusters[i]) - 1][0])
    return gaps


def absolute_value(n):
    if n >= 0:
        return n
    else:
        return n * - 1


def absolute_value_of_line(line):
    return_line = []
    for point in line:
        return_line.append(absolute_value(point))
    return return_line


def get_forward_numerical_first_order_gradient(points):
    gradient_line = []
    for i in range(len(points)):
        if i < (len(points) - 2):
            gradient_line.append(float(points[i + 1]) - float(points[i]))
    return gradient_line


def absolute_value_filtering(points, filter_value):
    filtered_points = []
    for point in points:
        if absolute_value(point) < filter_value:
            filtered_points.append(0)
        else:
            filtered_points.append(point)
    return filtered_points


def get_subline_of_line(line, start_index, end_index):
    sub_line = []
    for i in range(start_index, end_index + 1):
        sub_line.append(line[i])
    return sub_line


def get_median_function_of_line(line, median_reach):
    median_function_line = []
    for i in range(len(line)):
        if i < median_reach:
            median_function_line.append(statistics.median_low(get_subline_of_line(line, 0, i)))
        elif i > (len(line) - median_reach - 1):
            median_function_line.append(statistics.median_low(get_subline_of_line(line, i - median_reach, len(line) - 1)))
        else:
            median_function_line.append(statistics.median_low(get_subline_of_line(line, i - median_reach, i + median_reach)))
    return median_function_line


def average(line):
    return sum(line)/len(line)


def get_average_function_of_line(line, average_reach):
    average_function_line = []
    for i in range(len(line)):
        if i < average_reach:
            average_function_line.append(average(get_subline_of_line(line, 0, i)))
        elif i > (len(line) - average_reach - 1):
            average_function_line.append(average(get_subline_of_line(line, i - average_reach, len(line) - 1)))
        else:
            average_function_line.append(average(get_subline_of_line(line, i - average_reach, i + average_reach)))
    return average_function_line


def add_constant_to_line(line, constant):
    new_line = []
    for point in line:
        new_line.append(float(point) + constant)
    return new_line


def multiply_line_by_constant(line, constant):
    new_line = []
    for point in line:
        new_line.append(point * constant)
    return new_line


def get_copy_of_line(line):
    new_line = []
    for point in line:
        new_line.append(point)
    return new_line


def get_peak_analysis_of_line(line, average_filter_range=30, linear_transform_alpha=1.8, linear_transform_beta=2, rain_detection_minimal_first_gradient=15, is_standart_video=True):
    np_line = np.array(line)

    # Frist Gradient
    np_line_first_gradient = np.delete(np.convolve(np_line, np.array([1, -1])), 0)

    # Second Gradient
    np_line_second_gradient = np.delete(np.convolve(np_line_first_gradient, np.array([1, -1])), 0)

    # Absolute Value
    np_line_second_gradient_abs = np.abs(np_line_second_gradient)

    # Local Average Filtering
    np_line_second_gradient_abs_avg_filterd = np.array(get_average_function_of_line(np_line_second_gradient_abs.tolist(), average_filter_range))

    # Linear transformation of line
    np_line_second_gradient_abs_avg_filterd_transformed = np_line_second_gradient_abs_avg_filterd * linear_transform_alpha + linear_transform_beta

    # Get Peak Values
    peak_values = []
    for i in range(len(np_line_second_gradient_abs)):
        if np_line_second_gradient_abs[i] <= np_line_second_gradient_abs_avg_filterd_transformed[i]:
            peak_values.append(0)
        else:
            peak_values.append(1)

    # Get rain values via peak extension and peak connection
    rain_values = get_copy_of_line(peak_values)
    for i in range(len(rain_values)):
        if rain_values[i] == 1:
            # Peak extension left
            j = i - 1
            while j > 0 and not (rain_values[j] == 1) and (absolute_value(np_line_first_gradient[j] >= rain_detection_minimal_first_gradient)):
                rain_values[j] = 1
                j -= 1
            # Peak extension right
            j = i + 1
            while j < (len(rain_values)) and not (rain_values[j] == 1) and (absolute_value(np_line_first_gradient[j]) > rain_detection_minimal_first_gradient):
                rain_values[j] = 1
                j += 1

    if is_standart_video:
        # Peak combination
        for i in range(len(rain_values) - 4):
            if rain_values[i] == 1 and rain_values[i + 4] == 1:
                rain_values[i + 1] = 1
                rain_values[i + 2] = 1
                rain_values[i + 3] = 1
            elif rain_values[i] == 1 and rain_values[i + 3] == 1:
                rain_values[i + 1] = 1
                rain_values[i + 2] = 1
            elif rain_values[i] == 1 and rain_values[i + 2] == 1:
                rain_values[i + 1] = 1
        # Filter out single peaks
        for i in range(1, len(rain_values) - 1):
            if rain_values[i] == 1 and rain_values[i - 1] == 0 and rain_values[i + 1] == 0:
                rain_values[i] = 0

    # Return rain values
    return rain_values


def get_peak_detection_image(image):
    return_image = []
    for i in range(len(image)):
        return_image.append(get_peak_analysis_of_line(image[i]))
    return return_image


def sum_of_line(line):
    sum = 0
    for p in line:
        sum += p
    return sum

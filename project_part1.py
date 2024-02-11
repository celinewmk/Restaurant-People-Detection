import numpy as np
import cv2 as cv
import csv


def read_txt_file(txt_filename):
    data = []
    with open(txt_filename, newline="") as txtfile:
        reader = csv.reader(txtfile, delimiter=",")
        for row in reader:
            image_id, x, y, width, height = map(int, row)
            data.append((image_id, x, y, width, height))
    return data


def get_rectangle_using_coordinates(image, x, y, width, height):
    """
    Given an image, returns a new image that is only the rectangle portion
    specified by x, y, width, height.
    """
    return image[y : y + height, x : x + width]


def get_hsv_histogram(rectangle):
    """
    Given an image (expected to be the rectangle portion of the original image),
    returns its HSV histogram.
    """
    hsv = cv.cvtColor(rectangle, cv.COLOR_BGR2HSV)
    histogram = cv.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv.normalize(histogram, histogram).flatten()


def display_rectangle(rectangle):
    """
    Display rectangle (usually a person) on the screen for testing purposes.
    """
    cv.imshow("Image", rectangle)
    cv.waitKey(0)
    cv.destroyAllWindows()


def calculate_hist_img(image_filename, coordinates) -> list[tuple]:
    """
    Returns
    [
        (hist1_full, hist1_half), 
        (hist2_full, hist2_half),
        ...
    ]
    """
    image = cv.imread(image_filename)
    assert image is not None, "file could not be read, check with os.path.exists()"

    histograms = []
    for coord in coordinates:
        person_full = get_rectangle_using_coordinates(
            image, coord[0], coord[1], coord[2], coord[3]
        )  # full rectangle
        display_rectangle(person_full)
        person_half = get_rectangle_using_coordinates(
            image, coord[0], coord[1], coord[2], coord[3] // 2
        )  # half rectangle
        display_rectangle(person_half)
        histograms.append(
            (get_hsv_histogram(person_full), get_hsv_histogram(person_half))
        )  # i.e.: (HISTOGRAM_FULL, HISTOGRAM_HALF)

    return histograms


if __name__ == "__main__":
    # txt_filename = "labels.txt"  # Replace with the actual filename
    # data = read_txt_file(txt_filename)
    # print(data[0][0])
    test_image_filenames = ["1636738315284889400.png", "1636738357390407600.png"]
    image1_coords = [(232, 128, 70, 269), (375, 271, 156, 188), (333, 136, 85, 219)]
    # image2_coords = [(463, 251, 112, 206), (321, 269, 123, 189)]

    histograms1 = calculate_hist_img(test_image_filenames[0], image1_coords)
    comparison = cv.compareHist(histograms1[0][0], histograms1[1][0], cv.HISTCMP_CORREL)
    print(comparison)
    comparison = cv.compareHist(histograms1[0][0], histograms1[1][0], cv.HISTCMP_CORREL)
    print(comparison)
    print("========================================================")
    # # histograms2 = calculate_hist_img(test_image_filenames[1], image2_coords)
    # # histograms1 = calculate_hist_top_half_img(test_image_filenames[0], image1_coords)
    # print("========================================================")
    # histograms2 = calculate_hist_top_half_img(test_image_filenames[1], image2_coords)

    # # print("Compare hist1 with hist2")
    # # comparison = cv.compareHist(histograms2[0], histograms2[1], 0)
    # # print(comparison)

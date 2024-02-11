import numpy as np
import cv2 as cv


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


def calculate_hist_img(image_filename, coordinates):
    image = cv.imread(image_filename)  # image 1
    assert image is not None, "file could not be read, check with os.path.exists()"

    people = []
    histograms = []
    for coord in coordinates:
        person = get_rectangle_using_coordinates(
            image, coord[0], coord[1], coord[2], coord[3]
        )
        people.append(person)
        histograms.append(get_hsv_histogram(person))

    return histograms


if __name__ == "__main__":
    image_filenames = ["1636738315284889400.png", "1636738357390407600.png"]
    image1_coords = [(232, 128, 70, 269), (375, 271, 156, 188), (333, 136, 85, 219)]
    image2_coords = [(463, 251, 112, 206), (321, 269, 123, 189)]

    histograms1 = calculate_hist_img(image_filenames[0], image1_coords)
    print("========================================================")
    histograms2 = calculate_hist_img(image_filenames[1], image2_coords)

    print("Compare hist1 with hist2")
    comparison = cv.compareHist(histograms2[0], histograms2[1], 0)
    print(comparison)

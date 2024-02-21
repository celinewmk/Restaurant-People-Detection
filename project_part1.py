import numpy as np
import cv2 as cv
import csv
import os

def read_image_if_exists(file_path):
    return cv.imread(file_path) if os.path.exists(file_path) else None

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
    histogram = cv.calcHist([hsv], [0], None, [256], [0, 256])
    return cv.normalize(histogram, histogram).flatten()


def display_rectangle(rectangle):
    """
    Display rectangle (usually a person) on the screen for testing purposes.
    """
    cv.imshow("Image", rectangle)
    cv.waitKey(0)
    cv.destroyAllWindows()


def calculate_hist_img(image_filename: str, coordinates: list[tuple]) -> list[dict]:
    """
    Calculates the histograms of rectangles generated from a give list of coordinates.

    Args:
        image_filename: string filename/path of the image file to be processed
        coordinates: list of coordinates of all rectangles in tuple format (X, Y, width, height)

    Returns:
        List of full and half histograms for each rectangle
    """
    image = read_image_if_exists(image_filename)
    # assert image is not None, "file could not be read, check with os.path.exists()"
    if image is None:
        return

    histograms: list[dict] = []
    for coord in coordinates:
        person_full = get_rectangle_using_coordinates(
            image, coord[0], coord[1], coord[2], coord[3]
        )  # full rectangle
        # display_rectangle(person_full)
        person_half = get_rectangle_using_coordinates(
            image, coord[0], coord[1], coord[2], coord[3] // 2
        )  # half rectangle
        # display_rectangle(person_half)
        histograms.append(
            {
                "full": get_hsv_histogram(person_full),
                "half": get_hsv_histogram(person_half),
            }
        )

    return histograms


def read_text_file(file_path):
    """
    Returns
    {
        "filename1": [(X, Y, W, H), (X, Y, W, H), (X, Y, W, H)]
    }
    """
    big_list = {}
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            filename = parts[0]
            values = tuple(map(int, parts[1:]))
            if filename not in big_list:
                big_list[filename] = []
            if values[3] > 50:
                big_list[filename].append(values)
    return big_list


if __name__ == "__main__":
    # txt_filename = "labels.txt"  # Replace with the actual filename
    # data = read_txt_file(txt_filename)
    # print(data[0][0])
    test_image_filenames = ["1636738315284889400.png", "1636738357390407600.png"]
    image1_coords = [(232, 128, 70, 269), (375, 271, 156, 188), (333, 136, 85, 219)]

    # Read labels.txt file into list
    label_file: dict = read_text_file("labels.txt")
    print(label_file["1636738315831747000"])

    # Calculate histogram of everyone in image 1
    histograms1 = calculate_hist_img(test_image_filenames[0], image1_coords)

    person1_img1 = histograms1[0]  # person 1
    top_labels_person1 = []  # (filename, X, Y, W, H, comparison_value) top 100

    # Loop through labels.txt file to compare person 1 in image 1 with label
    for image_name, values in label_file.items():
        # image_name: 100000202020
        # values = [(463, 251, 112, 206), (321, 269, 123, 189)] 0, 1
        current_histograms = calculate_hist_img(
            f"subsequence_cam1/1636738315831747000.png",
            [(334, 135, 105, 243), (247, 136, 89, 270), (380, 272, 143, 180)],
        )

        max_comparison = []

        for current_hist in current_histograms:
            # current_hist is of the full and the half rectangle

            full_full = cv.compareHist(
                person1_img1["full"], current_hist["full"], cv.HISTCMP_CORREL
            )
            full_half = cv.compareHist(
                person1_img1["full"], current_hist["half"], cv.HISTCMP_CORREL
            )
            half_full = cv.compareHist(
                person1_img1["half"], current_hist["full"], cv.HISTCMP_CORREL
            )
            half_half = cv.compareHist(
                person1_img1["half"], current_hist["half"], cv.HISTCMP_CORREL
            )
            max_comparison.append(max(full_full, full_half, half_full, half_half))

        print(max_comparison)
        break

    # Need to associate histogram with rectangle/label

    # image2_coords = [(463, 251, 112, 206), (321, 269, 123, 189)]

    # histograms1 = calculate_hist_img(test_image_filenames[0], image1_coords)
    # comparison = cv.compareHist(
    #     histograms1[0]["full"], histograms1[1]["full"], cv.HISTCMP_CORREL
    # )
    # print(comparison)
    # comparison = cv.compareHist(
    #     histograms1[0]["full"], histograms1[0]["half"], cv.HISTCMP_CORREL
    # )
    # print(comparison)

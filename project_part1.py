import numpy as np
import cv2 as cv
import csv
import os


def read_image_if_exists(file_path: str) -> cv.typing.MatLike | None:
    """
    Returns the image read using OpenCV's imread, if the file exists.
    If not, returns None.
    """
    return cv.imread(file_path) if os.path.exists(file_path) else None


def get_rectangle_using_coordinates(
    image: cv.typing.MatLike, x: int, y: int, width: int, height: int
) -> cv.typing.MatLike:
    """
    Given an image, returns a new image that is only the rectangle portion
    specified by x, y, width, height.
    """
    return image[y : y + height, x : x + width]


def get_normalized_hsv_histogram(rectangle: cv.typing.MatLike) -> cv.typing.MatLike:
    """
    Given an image (expected to be the rectangle portion of the original image),
    returns its normalized HSV histogram.
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


def calculate_hist_img(image_filename: str, coordinates: list[tuple]) -> list[dict] | None:
    """
    Calculates the HSV histograms of rectangles generated from a given list of coordinates.

    Args:
        image_filename: string filename/path of the image file to be processed
        coordinates: list of coordinates of all rectangles in tuple format (X, Y, width, height)

    Returns:
        List of full and half histograms for each rectangle
    """
    image = read_image_if_exists(image_filename)
    # assert image is not None, "file could not be read, check with os.path.exists()"
    if image is None:
        return None

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
                "full": get_normalized_hsv_histogram(person_full),
                "half": get_normalized_hsv_histogram(person_half),
            }
        )

    return histograms


def read_text_file(file_path) -> dict[str, list[tuple[int]]]:
    """
    Returns
    {
        "filename1": [(X, Y, W, H), (X, Y, W, H), (X, Y, W, H)],
        "filename2": [(X, Y, W, H), (X, Y, W, H)],
        ...
    }
    """
    big_list: dict[str, list[tuple[int]]] = {}
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

    test_image_filenames = ["1636738315284889400.png", "1636738357390407600.png"]
    image1_coords = [(232, 128, 70, 269), (375, 271, 156, 188), (333, 136, 85, 219)]

    # Read labels.txt file into list
    label_file = read_text_file("labels.txt")
    # print(label_file["1636738315831747000"])

    # Calculate histogram of everyone in image 1
    histograms1 = calculate_hist_img(test_image_filenames[0], image1_coords)

    person1_img1 = histograms1[0]  # person 1
    top_labels_person1 = []  # (filename, X, Y, W, H, comparison_value) top 100

    # Loop through labels.txt file to compare hists of person 1 in image 1 with hists of labels
    for image_name, values in label_file.items():
        # test values
        # image_name = 1636738315831747000
        # values = [(334, 135, 105, 243), (247, 136, 89, 270), (380, 272, 143, 180)]
        current_histograms = calculate_hist_img(
            f"subsequence_cam1/{image_name}.png", values
        )

        if current_histograms is None or len(current_histograms) == 0:
            continue

        max_comparison = []

        for current_hist in current_histograms:
            # current_hist is the full and half histograms of the current rectangle
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

        ### testing with hardcoded values ##
        max_in_image = max(max_comparison)
        # index_of_max = max_comparison.index(max_in_image)
        coord_of_max = values[max_comparison.index(max_in_image)]

        #  code to verify the person: uncomment code below if u want to see the image
        # image = read_image_if_exists("subsequence_cam1/1636738315831747000.png")
        # image = read_image_if_exists(f"subsequence_cam1/{image_name}.png")
        # person_found = get_rectangle_using_coordinates(
        #     image,
        #     coord_of_max[0],
        #     coord_of_max[1],
        #     coord_of_max[2],
        #     coord_of_max[3],
        # )
        # display_rectangle(person_found)

        # adding in top_labels_person1 as format (filename, X, Y, W, H, comparison_value)
        top_labels_person1.append(
            (
                image_name,
                coord_of_max[0],  # x
                coord_of_max[1],  # y
                coord_of_max[2],  # width
                coord_of_max[3],  # height
                max_in_image,  # float value of HSV comparison with test image
            )
        )

        # break

    # print(top_labels_person1)
    # for label in top_labels_person1:
    #     print(label)

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

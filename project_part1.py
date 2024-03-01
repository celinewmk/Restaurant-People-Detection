import cv2 as cv
import os


def read_image_if_exists(file_path: str) -> cv.typing.MatLike:
    """
    Reads an image file given its path.

    Args:
        file_path: path to image file

    Returns:
        The OpenCV-parsed image, if the file exists. Otherwise, returns None.
    """
    return cv.imread(file_path) if os.path.exists(file_path) else None


def get_rectangle_using_coordinates(
    image: cv.typing.MatLike, x: int, y: int, width: int, height: int
) -> cv.typing.MatLike:
    """
    Given an image, returns a new image that is only the rectangle cutout
    portion specified by x, y, width, height.

    Args:
        image: an image parsed by OpenCV's imread function
        x: x integer coordinate of the image's top left corner
        y: y integer coordinate of the image's top left corner
        width: integer value denoting the image's horizontal width
        height: integer value denoting the image's vertical height

    Returns:
        An OpenCV-parsed image.
    """
    return image[y : y + height, x : x + width]


def get_normalized_hsv_histogram(rectangle: cv.typing.MatLike) -> cv.typing.MatLike:
    """
    Given an OpenCV-parsed image (expected to be the rectangle portion of the original image),
    returns its normalized HSV histogram.

    We normalize the histogram so that they can be compared in terms of their distributions,
    regardless of factors such as image size, brightness, or overall colour intensity.

    For instance, a red pixel could be highly saturated in one image, but not as saturated
    in another image, due to various differences in size and brightness. By normalizing
    the histogram, we make remove those differences to make the comparisons more accurate.

    Args:
        rectangle: an image parsed by OpenCV's imread function

    Returns:
        The normalized HSV histogram of the given image.
    """
    hsv = cv.cvtColor(rectangle, cv.COLOR_BGR2HSV)
    histogram = cv.calcHist([hsv], [0], None, [256], [0, 256])
    return cv.normalize(histogram, histogram).flatten()


def display_rectangle(image: cv.typing.MatLike, image_title: str = "Image"):
    """
    Displays an image on a new window for visualization and testing.

    Example: Can be used to view a rectangle cutout of a person.

    Args:
        An OpenCV-parsed image.
    """
    cv.imshow(image_title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def calculate_hist_img(image_filename: str, coordinates: list[tuple]) -> list[dict]:
    """
    Calculates the HSV histograms of rectangles generated from a given list of coordinates.

    Args:
        image_filename: string filename/path of the image file to be processed
        coordinates: list of coordinates of all rectangles in tuple format (X, Y, width, height)

    Returns:
        List of full and half histograms for each rectangle. None if the image does not exist.
    """
    image = read_image_if_exists(image_filename)
    if image is None:
        return None

    histograms: list[dict] = []
    for coord in coordinates:
        person_full = get_rectangle_using_coordinates(
            image, coord[0], coord[1], coord[2], coord[3]
        )  # full rectangle
        person_half = get_rectangle_using_coordinates(
            image, coord[0], coord[1], coord[2], coord[3] // 2
        )  # half rectangle
        histograms.append(
            {
                "full": get_normalized_hsv_histogram(person_full),
                "half": get_normalized_hsv_histogram(person_half),
            }
        )

    return histograms


def read_text_file(file_path: str) -> dict[str, list[tuple[int]]]:
    """
    Reads the text file containing labels representing the rectangle cutouts
    of a person in an image, parses the file and returns a custom dictionary
    containing the same data in the text file.

    Format of the dictionary:
    ```
    {
        "filename1": [(X, Y, W, H), (X, Y, W, H), (X, Y, W, H)],
        "filename2": [(X, Y, W, H), (X, Y, W, H)],
        ...
    }
    ```

    Args:
        file_path: path to the labels text file

    Returns:
        The contents of the labels text file as a custom dictionary.
    """
    big_list: dict[str, list[tuple[int]]] = {}
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            filename = parts[0]

            # Ignores the two tests images found in the labels.txt file
            if filename != "1636738315284889400" and filename != "1636738357390407600":
                values = tuple(map(int, parts[1:]))
                if filename not in big_list:
                    big_list[filename] = []

                # Only append images whose heights are sufficiently large (i.e. 130 pixels)
                if values[3] > 130:
                    big_list[filename].append(values)
    return big_list


def save_results_to_text(data: list, person_name: str):
    """
    Creates a "results" folder in the project directory and saves the resulting
    100 best matching frame images into a text file stored in the folder.

    Args:
        data: resulting list of the best 100 matching frames after HSV histogram comparison
    """
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    with open(os.path.join(results_folder, f"output_{person_name}.txt"), "w") as file:
        # Loop through the array
        for element in data:
            # Write each element followed by a newline character
            file.write(f"{element}\n")


def save_100_images(best_100_matches: list, person_name: str):
    """
    Creates a "results" folder in the project directory and saves the resulting
    100 best matching frame images as PNG files stored in the folder.

    Args:
        data: resulting list of the best 100 matching frames after HSV histogram comparison
    """
    output_folder = os.path.join("results", str(person_name))
    os.makedirs(output_folder, exist_ok=True)

    for i in range(len(best_100_matches)):
        image_name = best_100_matches[i][0]
        image = read_image_if_exists(f"subsequence_cam1/{image_name}.png")
        person_found = get_rectangle_using_coordinates(
            image,
            best_100_matches[i][1],
            best_100_matches[i][2],
            best_100_matches[i][3],
            best_100_matches[i][4],
        )

        # Save the rectangle as a PNG image
        output_file = os.path.join(output_folder, f"{image_name}.png")
        cv.imwrite(output_file, person_found)


def find_100_best_matches(person: list[dict], person_name: str):
    """
    Finds the 100 image frames from 1000+ test image files (8600+ labelled frames) that 
    best match a given person's test image frame. Saves the results to a results folder.

    Args:
        data: resulting list of the best 100 matching frames after HSV histogram comparison
    """
    top_labels_person = []  # (filename, X, Y, W, H, comparison_value) top 100
    for image_name, values in label_file.items():

        current_histograms = calculate_hist_img(
            f"subsequence_cam1/{image_name}.png", values
        )

        if current_histograms is None or len(current_histograms) == 0:
            continue

        max_comparison = []

        for current_hist in current_histograms:
            # current_hist is the full and half histograms of the current rectangle
            full_full = cv.compareHist(
                person["full"], current_hist["full"], cv.HISTCMP_CORREL
            )
            full_half = cv.compareHist(
                person["full"], current_hist["half"], cv.HISTCMP_CORREL
            )
            half_full = cv.compareHist(
                person["half"], current_hist["full"], cv.HISTCMP_CORREL
            )
            half_half = cv.compareHist(
                person["half"], current_hist["half"], cv.HISTCMP_CORREL
            )
            max_comparison.append(max(full_full, full_half, half_full, half_half))

        max_in_image = max(max_comparison)
        coord_of_max = values[max_comparison.index(max_in_image)]

        # adding in top_labels_person1 as format (filename, X, Y, W, H, comparison_value)
        top_labels_person.append(
            (
                image_name,
                coord_of_max[0],  # x
                coord_of_max[1],  # y
                coord_of_max[2],  # width
                coord_of_max[3],  # height
                max_in_image,  # float value of HSV comparison with test image
            )
        )

    sorted_data = sorted(top_labels_person, key=lambda x: x[5], reverse=True)
    best_100_matches = sorted_data[:100]
    save_results_to_text(best_100_matches, person_name)
    save_100_images(best_100_matches, person_name)


if __name__ == "__main__":

    test_image_filenames = ["1636738315284889400.png", "1636738357390407600.png"]
    # Read labels.txt file into list
    label_file = read_text_file("labels.txt")

    image1_coords = [(232, 128, 70, 269), (375, 271, 156, 188), (333, 136, 85, 219)]
    image2_coords = [(463, 251, 112, 206), (321, 269, 123, 189)]

    # Calculate histogram of everyone in image
    histograms1 = calculate_hist_img(test_image_filenames[0], image1_coords)
    histograms2 = calculate_hist_img(test_image_filenames[1], image2_coords)

    # image 1
    print(f"======================= Image 1 ====================")
    print(f"[+] Calculating results for person in black...")
    find_100_best_matches(
        histograms1[0], "person_in_black_img1"
    )  # person in black jacket

    print(f"[+] Calculating results for person with a cap...")
    find_100_best_matches(histograms1[1], "person_with_cap_img1")  # person with a cap

    print(f"[+] Calculating results for person in pink...")
    find_100_best_matches(
        histograms1[2], "person_in_pink_img1"
    )  # person in pink jacket

    print(
        f"The 100 best matches have been found for each person in image 1 (1636738315284889400.png).\nPlease see results folder"
    )

    # image 2
    print(f"======================= Image 2 ====================")
    print(f"[+] Calculating results for person in pink...")
    find_100_best_matches(
        histograms2[0], "person_in_pink_img2"
    )  # person in pink jacket

    print(f"[+] Calculating results for person in black...")
    find_100_best_matches(
        histograms2[1], "person_in_black_img2"
    )  # person in black jacket

    print(
        f"The 100 best matches have been found for each person in image 2 (1636738357390407600.png).\nPlease see results folder"
    )

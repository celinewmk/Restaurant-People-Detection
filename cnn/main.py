import cv2 as cv
import numpy as np
import os
from PIL import Image, ImageOps
from utils import model, tools
import torch


def read_image_if_exists(file_path: str) -> cv.typing.MatLike:
    """
    Reads an image file given its path.

    Args:
        file_path: path to image file

    Returns:
        The OpenCV-parsed image, if the file exists. Otherwise, returns None.
    """
    return cv.imread(file_path) if os.path.exists(file_path) else None


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


def calculate_hist_img_filename(image_filename: str) -> list[dict]:
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

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the coordinates for cropping (top half)
    start_row, start_col = 0, 0
    end_row, end_col = int(height / 2), width

    person_full = image
    person_half = image[start_row:end_row, start_col:end_col]

    # cv.imshow('Original Image', person_full)
    # cv.imshow('Cropped Image (Top Half)', person_half)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    histograms = {
        "full": get_normalized_hsv_histogram(person_full),
        "half": get_normalized_hsv_histogram(person_half),
    }

    return histograms


def calculate_hist_img(image: Image) -> list[dict]:
    """
    Calculates the HSV histograms of the full and half portions of the given image.

    Args:
        image: PIL image object to be processed

    Returns:
        List of full and half histograms for the image.
    """
    # Convert the PIL image to a numpy array
    image_np = np.array(image)

    # Get the dimensions of the image
    height, width = image_np.shape[:2]

    # Calculate the coordinates for cropping (top half)
    start_row, start_col = 0, 0
    end_row, end_col = int(height / 2), width

    # Crop the full and half portions of the image
    person_full = image_np
    person_half = image_np[start_row:end_row, start_col:end_col]

    # Calculate the histograms
    histograms = {
        "full": get_normalized_hsv_histogram(person_full),
        "half": get_normalized_hsv_histogram(person_half),
    }

    return histograms


def draw_bounding_boxes(image: np.ndarray, mask_tensor: torch.Tensor):
    """
    Draws bounding boxes around detected objects based on the mask tensor on the provided image.

    Args:
        image (numpy.ndarray): The original image as a NumPy array.
        mask_tensor (torch.Tensor): A binary mask tensor indicating the locations of objects.

    Returns:
        numpy.ndarray: The original image with bounding boxes drawn around detected objects.
    """
    # Ensure the mask is a NumPy array. If it's a tensor, convert it.
    if isinstance(mask_tensor, torch.Tensor):
        mask = mask_tensor.cpu().numpy()  # Ensure it's on CPU and convert to NumPy
    else:
        mask = mask_tensor

    # The mask might come in various shapes, e.g., (1, H, W), (H, W), or (H, W, C).
    # We need to ensure it's in the shape (H, W) for cv.findContours.
    if len(mask.shape) > 2:
        mask = mask.squeeze()  # Converts (1, H, W) or similar to (H, W).

    # Convert mask to binary in case it's not already. Assuming object pixels are > 0.
    mask_binary = np.where(mask > 0, 255, 0).astype(np.uint8)

    # Find contours from the binary mask
    contours, _ = cv.findContours(mask_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes from contours on the original image
    for contour in contours:
        # Calculate bounding box coordinates
        x, y, w, h = cv.boundingRect(contour)
        # Draw rectangle on the image
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green boxes

    return image


def fit_to_person(person_image):
    # Convert image to grayscale
    person_image_gray = person_image.convert("L")

    # Get bounding box of person
    bbox = person_image_gray.getbbox()

    if bbox:
        # Calculate width and height of bounding box
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # Create a new blank image with the same size as the bounding box
        new_image = Image.new("RGB", (width, height), color="black")

        # Paste the person image onto the new image, aligning it to the top-left corner
        new_image.paste(person_image.crop(bbox), (0, 0))

        return new_image
    else:
        # No person found in the image, return None
        return None


def extract_people_from_masks(original_image, masks) -> list[dict]:
    """
    Applies each mask to the original image and extracts the result as a new image.

    Args:
        original_image (numpy.ndarray): The original image from which to extract people.
        masks (list of numpy.ndarray): A list of masks, where each mask corresponds to a person.

    Returns:
        List of PIL.Image: A list of image objects, each containing a person extracted from the original image.
    """
    # Get OG image and all masks.
    # Apply each mask on the image and extracted each person out as their own image
    # We fit the size of each image to the height/width of each person
    # We removed the remaining black background from each image
    # We return the list of images, each representing a cutout of a person
    people_images: list[dict] = []

    for mask in masks:
        # Make sure the mask is binary and has the same dimensions as the original image
        mask_binary = mask[0].cpu().numpy() > 0.5
        mask_binary = np.repeat(mask_binary[:, :, np.newaxis], 3, axis=2)

        # Apply the mask to the original image
        person_cutout = np.where(mask_binary, original_image, 0)

        # Convert the cutout to a PIL image
        person_image = Image.fromarray(person_cutout.astype("uint8"))
        person_image = fit_to_person(person_image)

        # ----- Remove black background from image
        # Convert PIL image to NumPy array
        person_image = np.array(person_image)

        # Convert image to image gray
        tmp = cv.cvtColor(person_image, cv.COLOR_BGR2GRAY)

        # Applying thresholding technique
        _, alpha = cv.threshold(tmp, 0, 255, cv.THRESH_BINARY)

        # Using cv2.split() to split channels
        # of coloured image
        b, g, r = cv.split(person_image)

        # Making list of Red, Green, Blue
        # Channels and alpha
        rgba = [b, g, r, alpha]

        # Using cv2.merge() to merge rgba
        # into a coloured/multi-channeled image
        dst = cv.merge(rgba, 4)
        image = Image.fromarray(dst.astype("uint8"))

        # exit()

        people_images.append({"person_image": image, "mask": mask})

    return people_images

def save_100_images(best_100_matches: list, person_name: str):
    """
    Creates a "results" folder in the project directory and saves the resulting
    100 best matching frame images as PNG files stored in the folder.

    Args:
        data: resulting list of the best 100 matching frames after HSV histogram comparison
    """
    output_folder = os.path.join("examples/results/cam0/", f"{str(person_name)}_100_best")
    os.makedirs(output_folder, exist_ok=True)

    for match in best_100_matches:
        image = match[1]
        image_name = match[0]
   

        # Save the rectangle as a PNG image
        output_file = os.path.join(output_folder, image_name)

        image.save(
            output_folder
        )


def find_100_best_matches(person: list[dict], person_name: str, folder_name: str):
    """
    Finds the 100 image frames from 1000+ test image files (8600+ labelled frames) that
    best match a given person's test image frame. Saves the results to a results folder.

    Args:
        data: resulting list of the best 100 matching frames after HSV histogram comparison
    """

    source_path_dir = f"images/{folder_name}"
    output_path_dir = f"examples/output/{folder_name}"

    counter = 0
    top_100_best = []
    for file in os.listdir(source_path_dir):
        if file.endswith(".png"):
            counter += 1
            # image_name = file
            image_name = file
            print(f"===========================================================")
            print(f"Segmenting {counter}: {image_name}")

            # Charger le modèle et appliquer les transformations à l'image
            seg_model, transforms = model.get_model()

            # Ouvrir l'image et appliquer les transformations
            image_path = os.path.join(source_path_dir, image_name)
            image = Image.open(image_path)
            transformed_img = transforms(image)

            # Effectuer l'inférence sur l'image transformée sans calculer les gradients
            with torch.no_grad():
                output = seg_model([transformed_img])

            # Traiter le résultat de l'inférence
            # result = tools.process_inference(output, image)
            list_of_masks: list = tools.process_inference(output, image)
            people_imgs = extract_people_from_masks(image, list_of_masks)

            all_histograms: list = []

            for person_img in people_imgs:
                # person_img.show()
                all_histograms.append(calculate_hist_img(person_img["person_image"]))

            # Max comparison value of the test person with all other people in the image
            max_comparison = []

            for hist in all_histograms:
                full_full = cv.compareHist(
                    person["full"], hist["full"], cv.HISTCMP_CORREL
                )
                full_half = cv.compareHist(
                    person["full"], hist["half"], cv.HISTCMP_CORREL
                )
                half_full = cv.compareHist(
                    person["half"], hist["full"], cv.HISTCMP_CORREL
                )
                half_half = cv.compareHist(
                    person["half"], hist["half"], cv.HISTCMP_CORREL
                )
                max_comparison.append(max(full_full, full_half, half_full, half_half))

            print(f"MAX COMPARISON: {max_comparison}")
            max_in_image = max(max_comparison)
            print(f"-- {max_in_image}")

            best_person = people_imgs[max_comparison.index(max_in_image)]
            # best_person["mask"]
            # best_person["person_image"].show()

            # image.show()
            img_bounding_box_np = draw_bounding_boxes(
                np.array(image), best_person["mask"]
            )
            img_bounding_box = Image.fromarray(img_bounding_box_np.astype(np.uint8))
            if not os.path.exists(f"{output_path_dir}/{person_name}"):
                os.makedirs(f"{output_path_dir}/{person_name}")
            img_bounding_box.save(
                os.path.join(f"{output_path_dir}/{person_name}", image_name)
            )
            top_100_best.append((image_name,img_bounding_box, max_in_image))
            # result.show()
            # img_with_bounding_box.show()

        else:
            break

    sorted_data = sorted(top_100_best, key=lambda x: x[1], reverse=True)
    best_100_matches = sorted_data[:100]
    save_100_images(top_100_best, person_name)


if __name__ == "__main__":
    test_image_filenames = [
        "images/person_1.png",
        "images/person_2.png",
        "images/person_3.png",
        "images/person_4.png",
        "images/person_5.png",
    ]

    # Calculate histogram of everyone in image
    histograms = [
        calculate_hist_img_filename(test_image_filenames[0]),
        calculate_hist_img_filename(test_image_filenames[1]),
        calculate_hist_img_filename(test_image_filenames[2]),
        calculate_hist_img_filename(test_image_filenames[3]),
        calculate_hist_img_filename(test_image_filenames[4]),
    ]
    # find_100_best_matches(histograms[0], f"person_{1}", "cam1")
    for i in range(0, 5):
        print(f"[+] Calculating results for person {i+1}...")
        find_100_best_matches(histograms[i], f"person_{i+1}", "cam0")

    # print(f"[+] Calculating results for person 2...")
    # find_100_best_matches(histograms1[1], "person_with_cap_img1")

    # print(f"[+] Calculating results for person in pink...")
    # find_100_best_matches(
    #     histograms1[2], "person_in_pink_img1"
    # )  # person in pink jacket

    # print(
    #     f"The 100 best matches have been found for each person in image 1 (1636738315284889400.png).\nPlease see results folder"
    # )

    # print(f"======================= Image 2 ====================")
    # print(f"[+] Calculating results for person in pink...")
    # find_100_best_matches(
    #     histograms2[0], "person_in_pink_img2"
    # )  # person in pink jacket

    # print(f"[+] Calculating results for person in black...")
    # find_100_best_matches(
    #     histograms2[1], "person_in_black_img2"
    # )  # person in black jacket

    # print(
    #     f"The 100 best matches have been found for each person in image 2 (1636738357390407600.png).\nPlease see results folder"
    # )

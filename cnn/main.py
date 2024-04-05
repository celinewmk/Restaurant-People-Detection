import cv2 as cv
import numpy as np
import os
import torch
from PIL import Image
from utils import model, tools


def read_image_if_exists(file_path: str) -> cv.typing.MatLike:
    """
    Reads an image file given its path.

    Args:
        file_path: path to image file

    Returns:
        The OpenCV-parsed image, if the file exists. Otherwise, returns None.
    """
    return cv.imread(file_path) if os.path.exists(file_path) else None

def remove_png_extension(filename):
    if filename.endswith('.png'):
        return filename[:-4]
    else:
        return filename


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

def calculate_hist_img_filename(image_filename: str) -> dict[str, cv.typing.MatLike]:
    """
    Calculates the normalized full/half HSV histograms of a given image by its filename.

    Args:
        image_filename: string filename/path of the image file to be processed

    Returns:
        Dictionary of the full and half histogram of the image. None if the image does not exist.
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

    histograms = {
        "full": get_rgb_histogram(person_full),
        "half": get_rgb_histogram(person_half),
    }

    return histograms


def calculate_hist_img(image: Image) -> dict[str, cv.typing.MatLike]:
    """
    Calculates the normalized full/half HSV histograms of a given image.

    Args:
        image: PIL Image object to be processed

    Returns:
        Dictionary of the full and half histogram of the image. None if the image does not exist.
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
        "full": get_rgba_histogram(person_full),
        "half": get_rgba_histogram(person_half),
    }

    return histograms


def draw_bounding_boxes(image: np.ndarray, mask_tensor: torch.Tensor) -> np.ndarray:
    """
    Draws bounding boxes around detected objects based on the mask tensor on the provided image.

    Args:
        image (numpy.ndarray): The original image as a NumPy array.
        mask_tensor (torch.Tensor): A binary mask tensor indicating the locations of objects.

    Returns:
        numpy.ndarray: The original image with bounding boxes drawn around detected objects.
    """
    # Convert mask from numpy array to tensor if needed
    if isinstance(mask_tensor, torch.Tensor):
        mask = mask_tensor.cpu().numpy()  # Ensure it's on CPU and convert to NumPy
    else:
        mask = mask_tensor

    # Ensure mask is in the shape (H, W) for cv.findContours
    if len(mask.shape) > 2:
        mask = mask.squeeze()

    # Convert mask to binary assuming object pixels are > 0
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


def fit_to_person(person_image: Image) -> Image:
    """
    Fits the full original image size to the person's size.

    After only retaining the mask of the person and making everything else in the
    image turn black, we want to now fit the image to the person's height and width.
    This function returns a new image where it is fitted to the person's size.

    Args:
        person_image: image after keeping only the contents inside the mask

    Returns:
        Image fitted to the person's size.
    """
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
    # --- Steps ---
    # Get OG image and all masks.
    # Apply each mask on the image and extracted each person out as their own image
    # We fit the size of each image to the height/width of each person
    # We removed the remaining black background from each image
    # We return the list of images, each representing a cutout of a person
    people_images: list[dict] = []

    for mask in masks:
        # Make sure the mask is binary and has the same dimensions as the original image
        # mask_binary = mask[0].cpu().numpy() > 0.5
        # mask_binary = np.repeat(mask_binary[:, :, np.newaxis], 3, axis=2)
        mask_binary = mask > 0.5

        # Apply the mask to the original image
        # person_cutout = np.where(mask_binary, original_image, 0)
        person_cutout = original_image * mask_binary[:, :, np.newaxis]

        # Convert the cutout to a PIL image
        person_image = Image.fromarray(person_cutout.astype("uint8"))
        person_image = fit_to_person(person_image)

        image_height = person_image.height

        if image_height > 120:
            # ----- Remove black background from image
            # Convert PIL image to NumPy array
            person_image = np.array(person_image)

            # Convert image to image gray
            tmp = cv.cvtColor(person_image, cv.COLOR_BGR2GRAY)

            # Apply thresholding
            _, alpha = cv.threshold(tmp, 0, 255, cv.THRESH_BINARY)

            # Use cv2.split() to split channels of coloured image
            b, g, r = cv.split(person_image)

            # Make list of RGB Channels and alpha
            rgba = [b, g, r, alpha]

            # Merge rgba into a coloured/multi-channeled image
            dst = cv.merge(rgba, 4)
            image = Image.fromarray(dst.astype("uint8"))
            # image.show()

            people_images.append({"person_image": image, "mask": mask})

    return people_images


def save_100_images(best_100_matches: list, person_name: str):
    """
    Creates a "results" folder in the project directory and saves the resulting
    100 best matching frame images as PNG files stored in the folder.

    Args:
        best_100_matches: list of the best 100 matching frames after HSV histogram comparison
                          tuple(image_name, image, comparison_value)
        person_name: name of the person image (e.g. person_1)
    """
    output_folder = os.path.join("examples/100_best/cam0/", str(person_name))
    os.makedirs(output_folder, exist_ok=True)

    for match in best_100_matches:
        image: Image = match[1]
        image_name: str = match[0]

        # Save the rectangle as a PNG image
        output_file = os.path.join(output_folder, image_name)
        image.save(output_file)


# =======================================================================
def get_rgb_histogram(image):
    
    ranges = [0, 256, 0, 256, 0, 256]
    b, g, r = cv.split(image)   
    hist_rgb = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8], ranges, accumulate=False)
    cv.normalize(hist_rgb, hist_rgb, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    
    return hist_rgb

def get_rgba_histogram(image):
    
    ranges = [0, 256, 0, 256, 0, 256]
    b, g, r, a = cv.split(image)

    # Create a mask for opaque regions (where alpha is not 0)
    mask = (a > 0)

    # Apply the mask to the RGB channels
    b = b[mask]
    g = g[mask]
    r = r[mask]

    # Combine the histograms for each channel using the predefined bins and ranges
    hist_rgb = cv.calcHist([b, g, r], [0, 1, 2], None, [8, 8, 8], ranges, accumulate=False)
    
    # Normalize the histogram
    cv.normalize(hist_rgb, hist_rgb, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    
    return hist_rgb
# =======================================================================


def find_100_best_matches(person_hist: dict, person_name: str, folder_name: str):
    """
    Finds the 100 image frames from the given image frames that best match a given person's
    test image frame. Saves the results to a results folder.

    Args:
        person_hist: HSV histogram of the given person test image
        person_name: name of the person (e.g.: person_1)
        folder_name: name of the cam0, cam1 folders
    """

    source_path_dir = f"images/{folder_name}"
    output_path_dir = f"examples/output/{folder_name}"

    counter = 0
    # top_100_best = []
    for file in os.listdir(source_path_dir):
        if file.endswith(".png"):
            counter += 1
            image_name = file
            image_path = os.path.join(source_path_dir, image_name)
      
            image = Image.open(image_path)
            print(f"===========================================================")
            print(f"Segmenting {counter}: {image_name}")

            image_name_no_ext = remove_png_extension(image_name)

            result = tools.apply_saved_mask(image, image_name_no_ext, folder_name)
            # result.show()
            masks = tools.get_masks(image_name_no_ext, folder_name)

            people_imgs = extract_people_from_masks(image, masks)

            all_histograms: list = []

            for person_img in people_imgs:
                # person_img.show()
                all_histograms.append(calculate_hist_img(person_img["person_image"]))

            # Max comparison value of the test person with all other people in the image
            max_comparison = []

            for hist in all_histograms:
                full_full = cv.compareHist(
                    person_hist["full"], hist["full"], cv.HISTCMP_CORREL
                )
                full_half = cv.compareHist(
                    person_hist["full"], hist["half"], cv.HISTCMP_CORREL
                )
                half_full = cv.compareHist(
                    person_hist["half"], hist["full"], cv.HISTCMP_CORREL
                )
                half_half = cv.compareHist(
                    person_hist["half"], hist["half"], cv.HISTCMP_CORREL
                )
                max_comparison.append(max(full_full, full_half, half_full, half_half))

            # print(f"MAX COMPARISON: {max_comparison}")
            max_in_image = max(max_comparison)
            print(max_in_image)
            # print(f"-- {max_in_image}")

            if max_in_image > 0.85:

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
                # top_100_best.append((image_name, img_bounding_box, max_in_image))

        else:
            break

    # sorted_data = sorted(top_100_best, key=lambda x: x[2], reverse=True)
    # best_100_matches = sorted_data[:100]
    # save_100_images(best_100_matches, person_name)


if __name__ == "__main__":

    # test_image_filenames = [
    #     "images/person_1.png",
    #     "images/person_2.png",
    #     "images/person_3.png",
    #     "images/person_4.png",
    #     "images/person_5.png",
    # ]

    test_image_filenames = [
        "images/nobg_person_1.png",
        "images/nobg_person_2.png",
        "images/nobg_person_3.png",
        "images/nobg_person_4.png",
        "images/nobg_person_5.png",
    ]

    # test_images_after_mask: list = []
    # for img_filename in test_image_filenames:
    #     # Charger le modèle et appliquer les transformations à l'image
    #     seg_model, transforms = model.get_model()

    #     # Ouvrir l'image et appliquer les transformations
    #     image = Image.open(img_filename)
    #     transformed_img = transforms(image.convert("RGB"))

    #     # Effectuer l'inférence sur l'image transformée sans calculer les gradients
    #     with torch.no_grad():
    #         output = seg_model([transformed_img])

    #     masks = tools.process_inference(output, image)
    #     ppl = extract_people_from_masks(image, masks)
    #     # print(f"LEN PPLl {len(ppl)}")
    #     for person in ppl:
    #         person["person_image"].show()
    #     test_images_after_mask.append(ppl[0]["person_image"])
    #     # pic.show()
    # # exit()

    # histograms = [
    #     calculate_hist_img(test_images_after_mask[0]),
    #     calculate_hist_img(test_images_after_mask[1]),
    #     calculate_hist_img(test_images_after_mask[2]),
    #     calculate_hist_img(test_images_after_mask[3]),
    #     calculate_hist_img(test_images_after_mask[4]),
    # ]

    # Calculate histogram of all the people to detect
    histograms = [
        calculate_hist_img_filename(test_image_filenames[0]),
        calculate_hist_img_filename(test_image_filenames[1]),
        calculate_hist_img_filename(test_image_filenames[2]),
        calculate_hist_img_filename(test_image_filenames[3]),
        calculate_hist_img_filename(test_image_filenames[4]),
    ]
    find_100_best_matches(histograms[2], f"person_{3}", "cam0")

    # # cam0
    # for i in range(0, 5):
    #     print(f"[+] Calculating results for person {i+1}...")
    #     find_100_best_matches(histograms[i], f"person_{i+1}", "cam0")

    # # cam1
    # for i in range(0, 5):
    #     print(f"[+] Calculating results for person {i+1}...")
    #     find_100_best_matches(histograms[i], f"person_{i+1}", "cam1")

    


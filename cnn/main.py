import cv2 as cv
import numpy as np
import os
from PIL import Image
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


def calculate_hist_img(image_filename: str) -> list[dict]:
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

    histograms = (
        {
            "full": get_normalized_hsv_histogram(person_full),
            "half": get_normalized_hsv_histogram(person_half),
        }
    )

    return histograms

def draw_bounding_boxes(image, mask_tensor):
    # Convert torch tensor to numpy array if necessary
    mask = mask_tensor.numpy() if isinstance(mask_tensor, torch.Tensor) else mask_tensor

    # Ensure mask is binary (0 or 255)
    mask_binary = (mask > 0).astype(np.uint8) * 255

    # Check if the mask is empty
    if np.sum(mask_binary) == 0:
        print("Warning: Empty mask. Skipping contour detection.")
        return image

    # Find contours in the binary mask
    contours, _ = cv.findContours(mask_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes on the original image
    for contour in contours:
        # Get bounding box coordinates
        x, y, w, h = cv.boundingRect(contour)
        # Draw bounding box on the original image
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image



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
    
    for file in os.listdir(source_path_dir):
        if file.endswith(".png") and counter < 10:
            counter += 1

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
            result = tools.process_inference(output,image)
            
            max_comparison = []  # max comparison value of the test person with all other people in the image

            for mask in result:

                # Histogram of full mask
                histogram_full = cv.calcHist([np.array(mask)], [0], None, [256], [0,256])

                # Histogram of half mask
                height = mask.shape[0]
                # Slice the top half of the mask
                top_half_mask = mask[0:height//2, :]
                histogram_half = cv.calcHist([np.array(top_half_mask)], [0], None, [256], [0,256])
                
                full_full = cv.compareHist(
                    person["full"], histogram_full, cv.HISTCMP_CORREL
                )   
                full_half = cv.compareHist(
                    person["full"], histogram_half, cv.HISTCMP_CORREL
                )
                half_full = cv.compareHist(
                    person["half"], histogram_full, cv.HISTCMP_CORREL
                )
                half_half = cv.compareHist(
                    person["half"], histogram_half, cv.HISTCMP_CORREL
                )
                max_comparison.append(max(full_full, full_half, half_full, half_half))
            
            max_in_image = max(max_comparison)
        
            best_mask = result[max_comparison.index(max_in_image)]
            print(f"Mask ---- {best_mask}")
            print(type(best_mask))

                # Load the original image and the mask
            original_image = cv.imread(f"{source_path_dir}/{image_name}")

            # Assuming your mask is a torch.Tensor object named mask_tensor
            # Draw bounding boxes on the original image using the mask
            image_with_boxes = draw_bounding_boxes(original_image, best_mask)

            # Display the result
            cv.imshow('Image with Bounding Boxes', image_with_boxes)
            cv.waitKey(0)
            cv.destroyAllWindows()


            # result.save(os.path.join(output_path_dir, image_name))
            
            # result.show()
        else:
            break
    



if __name__ == "__main__":

    image1_coords = [(232, 128, 70, 269), (375, 271, 156, 188), (333, 136, 85, 219)]
    image2_coords = [(463, 251, 112, 206), (321, 269, 123, 189)]
    test_image_filenames = ["images/person_1.png", "images/person_2.png","images/person_3.png","images/person_4.png", "images/person_5.png"]

    # Calculate histogram of everyone in image
    histograms = [calculate_hist_img(test_image_filenames[0]), calculate_hist_img(test_image_filenames[1]),
    calculate_hist_img(test_image_filenames[2]), calculate_hist_img(test_image_filenames[3]), calculate_hist_img(test_image_filenames[4])]

    print(f"[+] Calculating results for person 1...")
    find_100_best_matches(
        histograms[0], "person_1", "cam0"
    )  

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


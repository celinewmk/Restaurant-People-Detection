import numpy as np
import cv2 as cv

def get_rectangle_using_coordinates(image, x, y, width, height):
    return image[y:y+height, x:x+width]

def get_hsv_histogram(rectangle):
    hsv = cv.cvtColor(rectangle, cv.COLOR_BGR2HSV)
    histogram = cv.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv.normalize(histogram, histogram).flatten()

if __name__ == "__main__":
    image = cv.imread('1636738315284889400.png')
    assert image is not None, "file could not be read, check with os.path.exists()"

    person1 = get_rectangle_using_coordinates(image,232, 128, 70, 269)
    person2 = get_rectangle_using_coordinates(image, 375, 271, 156, 188)
    person3 = get_rectangle_using_coordinates(image, 333, 136, 85, 219)

    histogram1 = get_hsv_histogram(person1)
    histogram2 = get_hsv_histogram(person2)
    histogram3 = get_hsv_histogram(person3)

    comparison = cv.compareHist(histogram1, histogram3, 0)
    print(comparison)


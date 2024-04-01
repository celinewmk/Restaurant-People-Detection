# Restaurant-People-Detection
A computer vision project utilizing image processing techniques implemented with Python and OpenCV to detect people's movements from a camera in a restaurant.

## Geting Started
Download the repository
- Have Python 3.x installed
- Clone the repository with Git
- Navigate to the root of the repository

Setup environment
- Create a virtual environment to install and manage dependencies
    - `python -m venv venv`
- Activate the virtual environment by running the activate script
    - `source venv/Scripts/activate` on Windows git bash
    - `source venv/lib/activate` on UNIX
- Install dependencies
    - `pip install -r requirements.txt`
- Execute the code
    - `python project_part1.py`

## Part 1: Bounding Box
In this first part of the project, we compare each person's HSV histogram in the two given test images `1636738315284889400.png` and `1636738357390407600.png` to the HSV histogram of every single person in each image frame in the `subsequence_cam1` folder. We measure the accuracy of our experiment by selecting the top 100 results by their HSV comparison value (1 being the highest, 0 being the lowest), and verifying our results by manually checking each resulting image against the person it is supposed to have recognized.

### Methodology
In part 1, we are given the file `labels.txt` containing a list of image file names (from the `subsequence_cam1` folder) and associated coordinates. Each coordinate represents the top-left corner of the cutout of a person in the image frame.

We are also given two test images `1636738315284889400.png` and `1636738357390407600.png`.

Our goal is to extract an image of each person in the two test images, by drawing two bounding boxes around each person.
- Bounding Box 1: Is an image that fits the entire person in the box
- Bounding Box 2: Is an image that fits the upper half of the person's body in the box

We do the same bounding box cutout method to every person in every image frame in the `subsequence_cam1` folder.

We calculate the histogram of each full/half cutout of each person in the two test images, to the histograms of every single full/half cutout of each person in every image in the `subsequence_cam1`.

Once completed, we take the top 100 results and place them in the results folder.

### Add test data (optional)
Steps to add the test data in the right places, if it is not already there.
- Place the two test images in the root of the project folder
    - Image 1: `1636738315284889400.png`
    - Image 2: `1636738357390407600.png`
- Place the `labels.txt` file in the root of the project folder
    - This file contains the labelled rectangle cutouts of the people in each frame in the video
- Place the `subsequence_cam1` folder in the root of the project folder
    - This folder contains the PNG images representing each frame of the video

Folder structure
- Red: Required input data
- Green: Expected output folder

![image](https://github.com/celinewmk/Restaurant-People-Detection/assets/67518620/2020ba63-497f-4390-b3de-5af86f85fd07)


## Part 2: Convolutional Neural Network (CNN)
In this part of the project, we are given the `images` folder which contains the starting data.

In it contains the frame-by-frame images from a security camera video footage recorded in a restaurant, where the folder `cam0` contains footage from camera 0, and `cam1` contains footage from camera 1.

We are also given 5 images, each image representing a person that was present in the security camera footages.

Our goal is to detect each of the 5 people the camera footage.

### Methodology
- Figure detection
    - First, we detect the exact figure of each person using a CNN model. This is much more accurate than using the bounding box method from part 1 of the project.
- HSV histogram conversion
    - Once we have the exact cutouts of each person, we represent their full body and the upper half of their body with normalized HSV histograms.
- Person detection
    - We detect each of the 5 given people in the camera footage frames by comparing their HSV histograms with the HSV histograms of every recognized figure/person in the camera footage.
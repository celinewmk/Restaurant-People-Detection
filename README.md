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

## Add test data (optional)
Steps to add the test data in the right places, if it is not already there.
- Place the two test images in the root of the project folder
    - Image 1: `1636738315284889400.png`
    - Image 2: `1636738357390407600.png`
- Place the `labels.txt` file in the root of the project folder
    - This file contains the labelled rectangle cutouts of the people in each frame in the video
- Place the `subsequence_cam1` folder in the root of the project folder
    - This folder contains the PNG images representing each frame of the video

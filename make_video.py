import cv2
import os
import re
from progress.bar import Bar

# A helper function to return digits inside text
def atoi(text):
    return int(text) if text.isdigit() else text

# A helper function to generate keys for sorting frames AKA natural sorting
def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def make_video(image_folder, video_name):
    images = [img for img in os.listdir(image_folder)]
    images.sort(key=natural_keys)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    video = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))
    bar = Bar('Creating Video', max=len(images))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        bar.next()

    cv2.destroyAllWindows()
    video.release()

    for file in os.listdir(image_folder):
        os.remove(image_folder + file)

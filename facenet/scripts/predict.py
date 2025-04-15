from keras_facenet import FaceNet
import os
import numpy as np
import cv2

# Load FaceNet embedder
embedder = FaceNet()

# Get paths from user
img1_path = input("Enter path of the base image: ")
folder_path = input("Enter path to the image directory to compare with: ")

# Read base image
img1 = cv2.imread(img1_path)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

def get_distance(img1, img2_path, threshold=1.5):
    img2 = cv2.imread(img2_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    emb1 = embedder.extract(img1)[0]['embedding']
    emb2 = embedder.extract(img2)[0]['embedding']

    dist = np.sum(np.square(emb1 - emb2))
    print(f"Distance with {img2_path}: {dist:.4f}")

    if dist <= threshold:
        print("-> SAME\n")
    else:
        print("-> DIFFERENT\n")

# Loop through folder and compare with base image
for filename in os.listdir(folder_path):
    img2_path = os.path.join(folder_path, filename)
    get_distance(img1, img2_path)

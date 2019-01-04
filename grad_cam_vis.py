from keras.models import load_model
from vis.visualization import visualize_cam, visualize_saliency
import argparse
import cv2
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-l","--layer", required=True, type=int,
	help="number of layer of filter to analize")
ap.add_argument("-f","--filter", required=False, type=int,
	default = None,
	help="number of filter to analize in the layer")
args = vars(ap.parse_args())

model = load_model(args["model"])
# image = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)
image = cv2.imread(args["image"])
# image = cv2.resize(image, (96, 96))
# ValueError: Cannot feed value of shape (1, 3, 89, 109) for Tensor 'conv2d_1_input:0', which has shape '(?, 109, 89, 3)'
# 109x89 to 89x109 (cambia x por y e y por x)
image = cv2.resize(image, (89, 109))
image = img_to_array(image)

heat_map = visualize_cam(model, args["layer"], None, image)
heat_map_saliency = visualize_saliency(model, args["layer"], None,seed_input=image)

plt.figure(0)
plt.title("Grad-CAM")
heat_map = cv2.resize(heat_map, (89*2, 109*2))
plt.imshow(heat_map)

# opencv2 channels are in order BGR. Transform the image to RGB to save it like a heatmap
# https://physiophile.wordpress.com/2017/01/12/why-opencv-uses-bgr-not-rgb/
destRGB = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)
destRGB = cv2.resize(destRGB, (89*3, 109*3))
cv2.imshow("image", destRGB)
# cv2.waitKey(0)

plt.figure(1)
plt.title("Saliency")
plt.imshow(heat_map_saliency)

plt.show()


"""
ver en que se esta fijando la red neuronal.
Cual es la parte mas relevante del rostro de una persona 
en la que se fija la red neuronal para distinguir entre hombre y mujer. Por qué
"""
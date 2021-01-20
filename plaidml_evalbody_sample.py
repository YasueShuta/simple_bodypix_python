import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from PIL import Image
# from utils import load_graph_model, get_input_tensors, get_output_tensors
from plaidml_utils import load_graph_model, get_input_tensors, get_output_tensors

# PATHS
imagePath = "C:/Users/shuta/Pictures/yasue_face.jpg"
modelPath = "bodypix_resnet50_float_model-stride16"

# CONSTRAINTS
OutputStride = 16

    
print("Loading model...", end="")
graph = load_graph_model(modelPath) # downloaded from the link above
print("done.\nLoading sample image...", end="")

def getBoundingBox(keypointPositions, offset=(10, 10, 10, 10)):
    minX = math.inf
    minY = math.inf
    maxX = - math.inf
    maxY = -math.inf
    for x, y in keypointPositions:
        if (x < minX):
            minX = x
        if(y < minY):
            minY = y
        if(x > maxX):
            maxX = x
        if (y > maxY):
            maxY = y
    return (minX - offset[0], minY-offset[1]), (maxX+offset[2], maxY + offset[3])


# load sample image into numpy array
img = tf.keras.preprocessing.image.load_img(imagePath)
imgWidth, imgHeight = img.size

targetWidth = (int(imgWidth) // OutputStride) * OutputStride + 1
targetHeight = (int(imgHeight) // OutputStride) * OutputStride + 1

print(imgHeight, imgWidth, targetHeight, targetWidth)
img = img.resize((targetWidth, targetHeight))
x = tf.keras.preprocessing.image.img_to_array(img, dtype=np.float32)
InputImageShape = x.shape
print("Input Image Shape in hwc", InputImageShape)


widthResolution = int((InputImageShape[1] - 1) / OutputStride) + 1
heightResolution = int((InputImageShape[0] - 1) / OutputStride) + 1
print('Resolution', widthResolution, heightResolution)

# Get input and output tensors
input_tensor_names = get_input_tensors(graph)
print(input_tensor_names)
output_tensor_names = get_output_tensors(graph)
print(output_tensor_names)
input_tensor = graph.get_tensor_by_name(input_tensor_names[0])

# Preprocessing Image
# For Resnet
if any('resnet_v1' in name for name in output_tensor_names):
    # add imagenet mean - extracted from body-pix source
    m = np.array([-123.15, -115.90, -103.06])
    x = np.add(x, m)
# For Mobilenet
elif any('MobilenetV1' in name for name in output_tensor_names):
    x = (x/127.5)-1
else:
    print('Unknown Model')
sample_image = x[tf.newaxis, ...]
print("done.\nRunning inference...", end="")

# evaluate the loaded model directly
with tf.compat.v1.Session(graph=graph) as sess:
    results = sess.run(output_tensor_names, feed_dict={
                       input_tensor: sample_image})
print("done. {} outputs received".format(len(results)))  # should be 8 outputs

import os
import numpy as np
import cv2
from PIL import Image
from utils import load_graph_model, get_input_tensors, get_output_tensors
import tensorflow as tf
from time import time
from io import StringIO
# make tensorflow stop spamming messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


# PATHS
# imagePath = "path/to/image"
# modelPath = "bodypix_resnet50_float_model-stride16"
modelPath = "bodypix_mobilenet_float_050_model-stride16"
skip_load_model = False
measure_time = True

# # CONSTANTS
OutputStride = 16

if not os.path.exists("test"):
    os.mkdir("test")

if not skip_load_model:
    print("Loading model...", end="")
    graph = load_graph_model(modelPath)  # downloaded from the link above
    print("done.")

if measure_time:
    timeFile = StringIO()
    timeFile.write(",Loop (ms), Session (ms), FPS (/sec)\n")



# load sample image from camera
cap = cv2.VideoCapture(0)
isOk, frame = cap.read()
print("Capture:", isOk)

height,width = 720, 1280
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 60)

img = Image.fromarray(frame)

assert img is not None
# img = tf.keras.preprocessing.image.load_img(imagePath)
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
def preprocess_image(x):
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
    return x[tf.newaxis, ...]

def get_input_from_frame(frame):
    img = Image.fromarray(frame)
    img = img.resize((targetWidth, targetHeight))
    x = tf.keras.preprocessing.image.img_to_array(img, dtype=np.float32)
    return preprocess_image(x), img

sample_image = preprocess_image(x)
print("done.\nRunning inference...", end="")
    
# open tensorflow session
sess = tf.compat.v1.Session(graph=graph)
# evaluate the loaded model directly
def eval_model(graph, input_):
    results = sess.run(output_tensor_names, feed_dict={
                        input_tensor: input_})
    return results
results = eval_model(graph, sample_image)
print("done. {} outputs received".format(len(results)))  # should be 8 outputs

for idx, name in enumerate(output_tensor_names):
    if 'displacement_bwd' in name:
        print('displacement_bwd', results[idx].shape)
    elif 'displacement_fwd' in name:
        print('displacement_fwd', results[idx].shape)
    elif 'float_heatmaps' in name:
        heatmaps = np.squeeze(results[idx], 0)
        print('heatmaps', heatmaps.shape)
    elif 'float_long_offsets' in name:
        longoffsets = np.squeeze(results[idx], 0)
        print('longoffsets', longoffsets.shape)
    elif 'float_short_offsets' in name:
        offsets = np.squeeze(results[idx], 0)
        print('offests', offsets.shape)
    elif 'float_part_heatmaps' in name:
        partHeatmaps = np.squeeze(results[idx], 0)
        print('partHeatmaps', partHeatmaps.shape)
    elif 'float_segments' in name:
        segIdx = idx
        segments = np.squeeze(results[idx], 0)
        print('segments', segments.shape)
    elif 'float_part_offsets' in name:
        partOffsets = np.squeeze(results[idx], 0)
        print('partOffsets', partOffsets.shape)
    else:
        print('Unknown Output Tensor', name, idx)



# Segmentation MASk
def get_segmentation_mask(results, segmentation_threshold=0.7):
    segments = np.squeeze(results[segIdx], 0)
    segmentScores = tf.sigmoid(segments)
    mask = tf.math.greater(segmentScores, tf.constant(segmentation_threshold))
    # print('maskshape', mask.shape)
    mask = tf.dtypes.cast(mask, tf.int32)
    ret = np.reshape(
        mask, (mask.shape[0], mask.shape[1]))
    # print('maskValue', segmentationMask[:][:])
    return ret


# Segmentation Threshold Trackbar
change_segmentation_threshold = True
dynamic_threshold = 0.7
def change_threshold(x):
    global dynamic_threshold
    dynamic_threshold = x / 100

if change_segmentation_threshold:
    winName = "Foreground Segmentation"
    cv2.namedWindow(winName)
    cv2.createTrackbar("Threshold (%)", winName, 0, 100, change_threshold)

# Load image from camera and calc segmentation
saveID = 1
count = 1
while True:
    start = time()
    isOk, frame = cap.read()
    if not isOk:
        continue
    cv2.imshow("Live", frame)

    # Create mask
    x, img = get_input_from_frame(frame)
    lap1 = time()
    results = eval_model(graph, x)
    lap2 = time()
    if change_segmentation_threshold:
        print("Th:", dynamic_threshold)
        mask = get_segmentation_mask(results, dynamic_threshold)
    else:
        mask = get_segmentation_mask(results)
    # resize_mask = mask * OutputStride
    # cv2.imshow("Mask", resize_mask)

    # Draw Segmented Output
    mask_img = Image.fromarray(mask * 255)
    mask_img = mask_img.resize(
        (targetWidth, targetHeight), Image.LANCZOS).convert("RGB")
    mask_img = tf.keras.preprocessing.image.img_to_array(
        mask_img, dtype=np.uint8)

    segmentationMask_inv = np.bitwise_not(mask_img)
    fg = np.bitwise_and(np.array(img), np.array(
        mask_img))
    cv2.imshow("Foreground Segmentation", fg)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("test/frame{}.jpg".format(saveID), frame)
        # cv2.imwrite("test/mask{}.jpg".format(saveID), resize_mask)
        cv2.imwrite("test/result{}.jpg".format(saveID), fg)
        saveID += 1

    done = time()
    if measure_time:
        loop_ms = (done - start) * 1000
        session_ms = (lap2 - lap1) * 1000
        fps = 1000 / loop_ms
        print("[{:04d}] Loop: {} ms, Session: {} ms, FPS: {}".format(
            count, int(loop_ms), int(session_ms), fps))
        print("{}, {}, {}, {}".format(
            count, int(loop_ms), int(session_ms), fps), file=timeFile)
        count += 1

# Closing
with open("test/measure_time.csv", "w") as f:
    f.write(timeFile.getvalue())
sess.close()
cap.release()
cv2.destroyAllWindows()

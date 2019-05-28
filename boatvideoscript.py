# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import glob

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_complete_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(graph=detection_graph, config = config)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed
video = cv2.VideoCapture('boatvideo.mp4')

frame_number = 0

while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    new_boxes = []
    new_scores = []
    new_classes = []
    new_num = []
    for i in range(len(classes[0])):
        if classes[0][i] == 9:
            new_boxes.append(boxes[0][i])
            new_scores.append(scores[0][i])
            new_classes.append(classes[0][i])
            #new_num.append(num[0][i])
        elif classes[0][i] == 1:
            new_boxes.append(boxes[0][i])
            new_scores.append(scores[0][i])
            new_classes.append(classes[0][i])
            #new_num.append(num[0][i])

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(new_boxes),
        np.squeeze(new_classes).astype(np.int32),
        np.squeeze(new_scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=0.20)
	
    print("It worked!")
    cv2.imwrite("boats/frame{}.png".format(frame_number), frame)
    frame_number += 1

    # All the results have been drawn on the frame, so it's time to display it.
    #cv2.imshow('Object detector', frame)
	
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        img_array = []
        for filename in glob.glob('boats/*.png'):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        out = cv2.VideoWriter('boats/boats.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        break

# Clean up
video.release()
cv2.destroyAllWindows()
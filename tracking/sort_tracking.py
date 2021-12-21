
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils

import torch
import time
import cv2
from sort import *

#create instance of SORT
mot_tracker = Sort() 
track_bbs_ids = []
frames = -1

CONFIDENCE = 0.95
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['background','beer','cola']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# PARSE YOUR MODEL HERE:
MODEL_PATH = "/home/andreasgp/MEGAsync/DTU/9. Semester/Deep Learning/object-tracking-project/02456-project/models/mobilenetv3_large_320_15epochs_entire_dataset.pth"

# PARSE YOUR VIDEO HERE:
VID_SOURCE = "/home/andreasgp/Documents/02456-project-21-12-21/02456-project/data/2021_10_28_12_49_00.avi" #livevideo1.MOV"#2021_10_28_12_49_00.avi"
#model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
model = torch.load(MODEL_PATH,map_location=torch.device('cpu'))
model = model.to(DEVICE)
model.eval()



# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")

vs = cv2.VideoCapture(VID_SOURCE)
#vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
fpsMax = 0
fpsMin = 100

# loop over the frames from the video stream
while True:
# grab the frame from the threaded video stream and resize it
# to have a maximum width of 400 pixels
    ret, frame = vs.read()
    frames += 1
    timer = cv2.getTickCount()
    if frame is None:
        break
    frame = imutils.resize(frame, width=480)
    orig = frame.copy()
    # convert the frame from BGR to RGB channel ordering and change
    # the frame from channels last to channels first ordering
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))
    # add a batch dimension, scale the raw pixel intensities to the
    # range [0, 1], and convert the frame to a floating point tensor
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    frame = torch.FloatTensor(frame)
    # send the input to the device and pass the it through the
    # network to get the detections and predictions
    frame = frame.to(DEVICE)
    with torch.no_grad():
        detections = model(frame)[0]
    rects = []
    valid_box = []

    # loop over the detections
    for i in range(0, len(detections["boxes"])):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections["scores"][i]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > CONFIDENCE:
            # extract the index of the class label from the
            # detections, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            rects.append(box.astype("int"))
            # draw the bounding box and label on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(orig, (startX, startY), (endX, endY),
            COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            valid_box.append(box)

	# update our centroid tracker using the computed set of bounding
	# box rectangles
    #print(detections)
    
    #print(valid_box)
    dets = []
    for box in valid_box:
        det = [box[0],box[1],box[2],box[3]]
        dets.append(det)
   # print(dets)
    #print(np.array(dets)[0,:])
    if len(dets)>0:
        track_bbs_ids = mot_tracker.update(np.array(dets))
#    print(track_bbs_ids)


	# loop over the tracked objects
    for i in range(len(track_bbs_ids)):
        #print(track_bbs_ids[i])
        centroid = track_bbs_ids[i,:4]
        #print(centroid)
        id = str(int(track_bbs_ids[i,-1]))
        #print(id)
        text = "ID {}".format(id)
        #print(text)
        #print(centroid[0] - 10, centroid[1] - 10)
        #cv2.putText(orig, text, (100+i*40,100),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(orig, text, (int(abs(centroid[0] - 10)), int(abs(centroid[1] - 10))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(orig, (int(centroid[0]), int(centroid[1])), 4, (0, 255, 0), -1)





    compute_time = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    if compute_time > fpsMax:
        fpsMax = compute_time
    if compute_time < fpsMin:
        fpsMin = compute_time
    #cv2.putText(orig, "FPS : " + str(float(compute_time)), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2);
    # show the output frame
    print("Frame:", frames)
    cv2.imshow("Frame", orig)
    key = cv2.waitKey(0) & 0xFF
    # if the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break
    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[INFO] interval FPS: [{:.2f} - {:.2f}]".format(fpsMin, fpsMax))
# do a bit of cleanup
cv2.destroyAllWindows()



# import the necessary packages
from centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2


def Main():
    ct = CentroidTracker()
    # load the COCO class labels
    LABELS = ["head", "person"]
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    # derive the paths to the YOLO weights and model configuration
    weightsPath = "model/yolov4-tiny-crowdhuman-416x416_final.weights"
    configPath = "model/yolov4-tiny-crowdhuman-416x416.cfg"
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)


    cap = cv2.VideoCapture("mall.mp4")

    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        print("[INFO] {} total frames in video".format(total))

    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    print(cap.isOpened())
    begin = time.time()
    left = 0
    while (cap.isOpened()):
        ret, image = cap.read()

        # load our input image and grab its spatial dimension

        if ret == True:
            (H, W) = image.shape[:2]

            # determine only the *output* layer names that we need from YOLO
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            # construct a blob from the input image and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes and
            # associated probabilities
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()

            # show timing information on YOLO
            print("[INFO] YOLO took {:.6f} seconds".format(end - start))

            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            boxes_c = []
            confidences = []
            classIDs = []
            rects = []
            trackableObject = {}
            total_person = 0
            left_people = 0
            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > 0.2 and classID == 1:
                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        # use the center (x, y)-coordinates to derive the top and
                        # and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        # update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        boxes_c.append([centerX - int(width/2), centerY - int(height/2), centerX + int(width/2), centerY + int(height/2)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.3)

            if len(idxs) > 0:
                for i in idxs.flatten():
                    rects.append(boxes_c[i])
            # update our centroid tracker using the computed set of bounding
            # box rectangles
            cv2.line(image, (W//2, 0), (W//2, H), (255, 0, 0), 2)
            objects = ct.update(rects)
            for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                
                #total objects of video
                total_person = max(total_person, objectID)
            
                #function to count overall people moving from left half to right half
                if centroid[0] > W//2:
                    left_people += 1
                    left = max(left, left_people)

            info = [
                ("crossing Left", left),
                ("Total people", total_person),
                ("Time taken", end - begin)
            ]

            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(image, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                    

            # show the output image

            cv2.imshow("Count and Track", image)
            # write the output frame to disk
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # release the file pointers
    print("[INFO] cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    finish = time.time()

    print(f"Total time taken : {finish - begin}")



if __name__ == "__main__":
    Main()

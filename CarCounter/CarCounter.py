from ultralytics import YOLO
import numpy as np
import cv2
import cvzone
from sort import *

model = YOLO("../Yolo-Weights/yolov8n.pt") #you can change here between v8n (nano), v8s (small), v8m (medium), v8l (large)
cap = cv2.VideoCapture("Videos/cars.mp4")
mask = cv2.imread("images/mask.png")

classNames = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
]

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [323, 350, 673, 350]

totalCount = set()

while True:
    succes, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    results = model(imgRegion, stream=True)

    imgGraphics = cv2.imread("images/graphics.png", cv2.IMREAD_UNCHANGED)

    cvzone.overlayPNG(imgBack=img, imgFront=imgGraphics, pos=(0, 0))

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Confidence
            conf = round(float(box.conf[0]), 2)
            # Class Name
            cls = box.cls[0]
            currClass = classNames[int(cls)]


            if currClass == "car" or currClass == "truck" or currClass == "motorbike" or currClass == "bus" and conf > 0.3:
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 3)
                # cvzone.putTextRect(img=img, text=f"{currClass} {conf}", pos=(max(0, x1), max(35, y1-20)), scale=0.7, thickness=1, offset=3)
                currArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currArray))

    resultsTracker = tracker.update(dets=detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for res in resultsTracker:
        x1, y1, x2, y2, ID = res
        w, h = x2-x1, y2-y1
        cx, cy = int(x1+w//2), int(y1+h//2)
        cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1]-30 < cy < limits[3] + 30:
            dl = len(totalCount)
            totalCount.add(ID)
            if len(totalCount) > dl:
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img=img, text=f"Count: {len(totalCount)}", pos=(50, 50))
    cv2.putText(img=img, text=str(len(totalCount)), org=(255, 100), color=(50, 50, 255), fontScale=5, fontFace=cv2.FONT_HERSHEY_PLAIN, thickness=8)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
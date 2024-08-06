from ultralytics import YOLO
import cv2
import cvzone

model = YOLO("../Yolo-Weights/yolov8n.pt")
cap = cv2.VideoCapture("Videos/cars.mp4")
mask = cv2.imread("masks/mask.png")

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

while True:
    succes, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
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
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 3)
                cvzone.putTextRect(img=img, text=f"{currClass} {conf}", pos=(max(0, x1), max(35, y1-20)), scale=0.7, thickness=1, offset=3)


    cv2.imshow("ImageRegion", img)
    cv2.waitKey(0)
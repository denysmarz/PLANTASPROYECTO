import cv2
from ultralytics import YOLO

# Initialize the webcam
cap = cv2.VideoCapture('20240423_132634.mp4')

# Initialize YOLOv7 object detector
model_path = ".\\bestyolol.onnx"
yolov8_detector = YOLO(model_path)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

# Define la nueva resolución
width = 640
height = 640

# Ajusta la resolución del video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Update object localizer
    results = yolov8_detector(frame,conf=0.4,vid_stride=50,stream_buffer=True)
    boxes = results[0].boxes.xyxy.cpu().numpy()
        # Obtener las etiquetas de las predicciones
    class_ids = results[0].boxes.cls.cpu().numpy()
    class_names = results[0].names
    #boxes, scores, class_ids = yolov8_detector(frame)
    # View results
    for box, class_id in zip(boxes, class_ids):
            
            x1, y1, x2, y2 = box[:4]
            cls = class_names[class_id]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(img_resized, f'{model.names[int(cls)]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, cls, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #cv2.putText(img_resized, int(conf), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Detected Objects', frame)


    #combined_img = yolov8_detector.draw_detections(frame)
    #cv2.imshow("Detected Objects", combined_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
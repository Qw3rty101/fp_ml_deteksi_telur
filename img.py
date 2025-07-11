from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/train10/weights/best.pt')

print(model.names)

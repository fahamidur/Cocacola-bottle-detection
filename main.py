import torch
import cv2
import numpy as np

confidence = 0
num_bottles = 0
# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')

# Load the input image
img= './test.jpeg'


results = model(img)
img1 = cv2.imread(img)


# Results
boxes = results.xyxy[0].numpy()
labels = results.names[0]
scores = results.xyxyn[0][:, 4].numpy()


for box, score in zip(boxes, scores):
   if score >= confidence:
    x1, y1, x2, y2, _ , _ = box
    x1, y1, x2, y2, _ , _ = box
    color = (0, 255, 0)
    thickness = 2
    # text = f"{labels}: {score:.2f}"
    cv2.rectangle(img1, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    # cv2.putText(img1, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    num_bottles = num_bottles + 1

cv2.putText(img1, f'Number of Bottles: {num_bottles}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow('Image', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
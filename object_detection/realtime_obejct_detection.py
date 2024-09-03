import sys
import signal

import torch
import torchvision
import cv2
import numpy as np

from object_detection.coco_classes import COCO_CLASSES


# Graceful exit
def signal_handler(sig, frame):
    print('Exiting gracefully...')
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
model.eval()  # Set the model to evaluation mode


def transform_image(image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    return transform(image)

def draw_boxes(image, boxes, labels):
    for box, label in zip(boxes, labels):
        named_label = COCO_CLASSES[label]
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(named_label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a tensor
    image_tensor = transform_image(frame)
    image_tensor = image_tensor.unsqueeze(0)

    # Perform object detection
    with torch.no_grad():
        outputs = model(image_tensor)

    # Extract bounding boxes and labels
    boxes = outputs[0]['boxes'].cpu().numpy().astype(np.int32)
    labels = outputs[0]['labels'].cpu().numpy()

    # Draw bounding boxes on the frame
    frame = draw_boxes(frame, boxes, labels)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


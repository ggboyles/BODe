# source myenv/bin/activate

# This is just some test code I've thrown together
# Make sure to create the venv with requirements.txt before trying to run

import cv2
from ultralytics import YOLO

# Loads YOLO model
model = YOLO("yolov8n.pt")   # this is a small and fast model

# Opens camera
camera = cv2.VideoCapture(1) # might needs switched to 0 depending on your computer

if not camera.isOpened():
    print("Camera not found")
    exit()

# Distance thresholds, can be tinkered with 
CLOSE_THRESHOLD = 120000
FAR_THRESHOLD = 30000

while True:
    success, frame = camera.read()

    if not success:
        print("Failed to read frame")
        break

    # Run object detection on this frame
    results = model(frame)

    # Variables to store closest object
    largest_area = 0
    best_box = None

    # Looks through all detected objects
    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            label = model.names[class_id]

            if label == "bottle":   # This can be changed to other objects, just have to look up what is supported by YOLO
            # label == "person" also works well, but distance thresholds need to be changed to work properly with that

                # Gets bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Calculates size (distance estimate)
                area = (x2 - x1) * (y2 - y1)

                # Keeps only the closest object, so BODe doesnt get confused
                if area > largest_area:
                    largest_area = area
                    best_box = (x1, y1, x2, y2)

    # If correct object, process it
    if best_box is not None:
        x1, y1, x2, y2 = best_box

        # Draws the rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if largest_area > CLOSE_THRESHOLD:
            # this is where the logic will go to stop BODe from moving forward
            message = "STOP"
            color = (0, 0, 255)   # Red

        elif largest_area < FAR_THRESHOLD:
            # Insert BODe moving forward code (at a good speed)
            message = "FOLLOW"
            color = (0, 255, 0)   # Green

        else:
            # Insert BODe moving forward code (at a slower speed)
            message = "GOOD DISTANCE"
            color = (0, 255, 255) # Yellow

        # Displays message
        cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Tracking", frame)

    # Press Q to quit out of screen
    if cv2.waitKey(1) == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()








# also need to factor in the four cameras, and what to do if detected in more than one
# at the same time

# we will want the object to be front and center most likely

# if detected on the left or right back cam, then BODe needs to rotate until 
# object is detected by front cams. This seems like it might be hard to do this smoothly
# but not sure.
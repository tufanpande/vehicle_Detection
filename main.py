import cv2
import numpy as np

# Open the video file or web camera
cap = cv2.VideoCapture('assets/video.mp4')

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Minimum dimensions for detected objects to be considered vehicles
min_width_rectangle = 80 
min_height_rectangle = 80

# Position of the line for counting vehicles
count_line_position = 550

# Initialize the background subtractor
algo = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

def center_handle(x, y, w, h):
    
    # Calculate the center of a bounding rectangle.
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []  # List to store detected vehicle centers
offset = 6   # Allowable error between pixels for detecting crossing
counter = 0  # Counter for the number of vehicles detected

while True:
    # Read a frame from the video
    ret, video = cap.read()
    
    # Check if frame is read correctly
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 5)

    # Apply background subtraction
    vid_sub = algo.apply(blur)
    
    # Perform dilation to fill in holes
    dilat = cv2.dilate(vid_sub, np.ones((5, 5)))
    
    # Define a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Apply morphological closing to remove small holes
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the processed frame
    countershape, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the counting line on the frame
    cv2.line(video, (25, count_line_position), (1200, count_line_position), (255, 0, 0), 3)

    for (i, c) in enumerate(countershape):
        # Get the bounding rectangle of each contour
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Check if the detected object meets the minimum size requirements
        val_counter = (w >= min_width_rectangle) and (h >= min_height_rectangle)
        if not val_counter:
            continue
        
        # Draw a rectangle around the detected vehicle
        cv2.rectangle(video, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(video, "Vehicle No: " + str(counter), (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2)

        # Calculate the center of the bounding rectangle
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(video, center, 4, (0, 0, 255), -1)

        # Check if the center of the detected vehicle has crossed the counting line
        for (x, y) in detect:
            if y < (count_line_position + offset) and y > (count_line_position - offset):
                counter += 1
                cv2.line(video, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
                detect.remove((x, y))
                print("Vehicle No: " + str(counter))

    # Display the current count of vehicles on the frame
    cv2.putText(video, "Vehicle No: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Show the processed video frame
    cv2.imshow('Detector', video)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

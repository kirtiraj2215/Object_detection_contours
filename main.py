import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Function required for the trackbar, it doesn't do anything
def nothing(x):
    pass

# Create a window for the color adjustments
cv2.namedWindow("Color Adjustments", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Color Adjustments", (300, 300))

# Create trackbars for adjusting the HSV lower and upper bounds
cv2.createTrackbar("Thresh", "Color Adjustments", 0, 255, nothing)

cv2.createTrackbar("lower_h", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("lower_s", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("lower_v", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("upper_h", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("upper_s", "Color Adjustments", 0, 255, nothing)
cv2.createTrackbar("upper_v", "Color Adjustments", 0, 255, nothing)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the frame for consistency
    frame = cv2.resize(frame, (400, 400))

    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Get the current positions of the trackbars for lower and upper HSV bounds
    l_h = cv2.getTrackbarPos("lower_h", "Color Adjustments")
    l_s = cv2.getTrackbarPos("lower_s", "Color Adjustments")
    l_v = cv2.getTrackbarPos("lower_v", "Color Adjustments")
    u_h = cv2.getTrackbarPos("upper_h", "Color Adjustments")
    u_s = cv2.getTrackbarPos("upper_s", "Color Adjustments")
    u_v = cv2.getTrackbarPos("upper_v", "Color Adjustments")
    
    # Define the lower and upper bounds for the HSV mask
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])
    
    # Create a mask that filters out everything except the specified HSV range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    filtr = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Invert the mask
    mask1 = cv2.bitwise_not(mask)
    # Get the threshold value from the trackbar
    m_g = cv2.getTrackbarPos("Thresh", "Color Adjustments")
    # Apply the threshold to create a binary image
    ret, thresh = cv2.threshold(mask1, m_g, 255, cv2.THRESH_BINARY)
    # Dilate the image to strengthen the contours
    dilata = cv2.dilate(thresh, (1, 1), iterations=6)
    
    # Find contours in the thresholded image
    cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop over each contour
    for c in cnts:
        epsilon = 0.0001 * cv2.arcLength(c, True)
        data = cv2.approxPolyDP(c, epsilon, True)
        
        # Find the convex hull for the contour
        hull = cv2.convexHull(data)
        # Draw the contour in blue
        cv2.drawContours(frame, [c], -1, (50, 50, 150), 2)
        # Draw the convex hull in green
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
    
    # Display the different processed images
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Mask", mask)
    cv2.imshow("Filter", filtr)
    cv2.imshow("Result", frame)
    
    # Break the loop if the user presses the 'ESC' key
    key = cv2.waitKey(25) & 0xFF
    if key == 27:
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
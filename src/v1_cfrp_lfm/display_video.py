import cv2 as cv
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'v1')
cap = cv.VideoCapture(os.path.join(DATA_DIR, 'cropped_cfrp_lfm.avi'))
empty = False

# while(not empty):
#     ret, frame = cap.read()
#     # np.set_printoptions(threshold=np.inf, linewidth=np.inf) # print without ellipses

#     if not ret:
#         empty = True
#     else:
#         gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         print(gray_frame)

# Display video
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was not grabbed, we have reached the end of the video
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv.imshow('Grayscale Video', gray_frame)

    # Press 'q' to exit the video display window
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv.destroyAllWindows()
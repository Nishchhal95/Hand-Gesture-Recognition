import cv2
import numpy as np
import os

GestureDictionary = {0 : "Peace",
                     1 : "Palm",
                     2 : "Fist",
                     3 : "Thumbs Up",
                     4 : "Ok"}
TOTAL_GESTURES = len(GestureDictionary)

# Create Directory Structure'
currentDir = "D:/Pyhton Projects/Python Gesture Recognition/"
if not os.path.exists(currentDir + "data"):
    os.makedirs(currentDir + "data")
    os.makedirs(currentDir + "data/train")
    os.makedirs(currentDir + "data/test")
    for x in range(TOTAL_GESTURES):
        os.makedirs(currentDir + "data/train/" + str(x))
        os.makedirs(currentDir + "data/test/" + str(x))

directory = currentDir + 'data/train/'
print(directory)

cap = cv2.VideoCapture(0)

while True:
    something, frame = cap.read()
    
    # Flips the Frame
    frame = cv2.flip(frame, 1)

    # Display Some Images count
    baseY = 120;
    incrementFactor = 20;
    for i in GestureDictionary:
        count = len(os.listdir(directory + str(i)))
        cv2.putText(frame, str(GestureDictionary[i]) + ": " + str(count), (10, baseY + (i * incrementFactor)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)  

    # Trying to create a Region of Interest
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])

    # Draw Region of Interest
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)

    # Get the Region of Interest
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64))

    # Show my Frame
    cv2.imshow("MyFrame", frame)

    # Handle Region of Interest
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    something, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY) # if the gray value is between 120 and 255 set to 1 otherwise it goes 0

    # Show Region of Interest
    cv2.imshow("Region of Interest", roi)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:
        break
    
    for x in GestureDictionary:
        if interrupt & 0xFF == ord(str(x)):
            path = directory + str(x) + '/'
            count = len(os.listdir(path))
            fileName = str(count) + '.jpg'
            print("Path : " + str(path))
            print("fileName : " + str(fileName))
            cv2.imwrite(path + fileName, roi)

cap.release()
cv2.destroyAllWindows()

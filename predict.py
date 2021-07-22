import numpy as np
from tensorflow.keras.models import model_from_json
import operator
import cv2
import sys, os

json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(model_json)
loaded_model.load_weights("model-bw.h5")
print("loaded model from Disk")

cap = cv2.VideoCapture(0)

while True:
    something, frame = cap.read()
    frame = cv2.flip(frame, 1)

    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])

    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)

    roi = frame[y1 : y2, x1 : x2]

    roi = cv2.resize(roi, (64, 64))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    something, testImage = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("Test", testImage)

    result = loaded_model.predict(testImage.reshape(1, 64, 64, 1))
    prediction = {'Peace': result[0][0],
                  'Palm': result[0][1],
                  'Fist': result[0][2],
                  'Thumbs Up': result[0][3],
                  'Ok': result[0][4]}

    predictionSorted = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

    cv2.putText(frame, predictionSorted[0][0], (x1 + 100, y2 + 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1)      
    cv2.putText(frame, str(int(prediction[predictionSorted[0][0]] * 100)) + "%", (x1 + 100, y2 + 60),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1)     
    cv2.imshow("Frame", frame)
    print("Result" + str(result))

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
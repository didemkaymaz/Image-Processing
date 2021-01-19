import cv2
import pandas as pd
from _datetime import datetime
import sys
import matplotlib as plt
import numpy as np
import plotly.express as px
import matplotlib as plot
import csv
import matplotlib.pyplot as plt


smile_cascade = cv2.CascadeClassifier('C:\\Users\\DK\\PycharmProjects\\GoruntuIslemeProje\\meter\\cascades\\smile_cascade.xml')
face_cascade = cv2.CascadeClassifier('C:\\Users\\DK\\PycharmProjects\\GoruntuIslemeProje\\meter\\cascades\\haarcascade_frontalface_default.xml')



times = []
smile_ratios = []
cap = cv2.VideoCapture('C:\\Users\\DK\\PycharmProjects\\GoruntuIslemeProje\\meter\\videos\\smile.mp4')


while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    a = str(len(faces))
    cv2.putText(img, 'Number of Faces Detected: ' + a, (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

        roi_gray = gray[y:y + h, x:x + w]
        roi_img = img[y:y + h, x:x + w]
        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=22, minSize=(25, 25))

        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_img, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            sm_ratio = str(round(sw / sh, 3))  #sm_ratio, gülümserken veya gülerken ağız genişliğinin ve yüksekliğinin oranını ifade eder.
            #Burada uygulanan mantık, bir kişi daha çok gülümsediğinde veya daha çok güldüğünde ağız genişliğinin artması,
            #buna karsılık sm_ratio'nun artmasıdır.
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Smile meter : ' + sm_ratio, (x + 5, y + h - 5), font, 1, (200, 255, 155), 2, cv2.LINE_AA)
            if float(sm_ratio) > 1.8:
                cv2.putText(img, "Smiling",  (x - 160, y + h - 160), cv2.FONT_HERSHEY_SIMPLEX, 2, 100, 2, cv2.LINE_AA)
            if float(sm_ratio) > 1.8:
                smile_ratios.append(float(sm_ratio))
                times.append(datetime.now())
    cv2.imshow('Smile Detector', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
ds= {'smile_ratios': smile_ratios, 'times': times}
df= pd.DataFrame(ds)
df.to_csv('smile_recordsVideo.csv')

df = pd.read_csv('C:\\Users\\DK\\PycharmProjects\\GoruntuIslemeProje\\meter\\smile_recordsVideo.csv')

fig = px.line(df, x='times', y='smile_ratios', title='Smile Records Video')

fig.show()


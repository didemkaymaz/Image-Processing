import cv2
import pandas as pd
from _datetime import datetime
import matplotlib as plt
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import csv
import sys

smile_cascade = cv2.CascadeClassifier('C:\\Users\\DK\\PycharmProjects\\GoruntuIslemeProje\\meter\\cascades\\smile_cascade.xml')
face_cascade = cv2.CascadeClassifier('C:\\Users\\DK\\PycharmProjects\\GoruntuIslemeProje\\meter\\cascades\\haarcascade_frontalface_default.xml')


img = cv2.imread('C:\\Users\\DK\\PycharmProjects\\GoruntuIslemeProje\\meter\\images\\woman.png')
#img = cv2.resize(img, (1150, 562))


times = []  #gulumseme zamanini saklar.
smile_ratios = [] #gulumseme oranini saklar



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gri tonlamalı görüntü", gray)
gray = cv2.GaussianBlur(gray, (21, 21), 0) #Gauss bulanıklığı işlevini uygulayarak görüntünün bazı ön işlemleri yapıldı.
faces = face_cascade.detectMultiScale(gray, 1.3, 5) #yüz kademeli sınıflandırıcının 'DetectMultiscale' işlevini kullanarak kullanıcı yüzünü tespit edilir.
# Algılandıktan sonra, burada yüz olan ve etrafına bir dikdörtgen çizen bir ROI (ilgi bölgesi) oluşturulur.
font = cv2.FONT_HERSHEY_SIMPLEX
a = str(len(faces))
cv2.putText(img, 'Number of Faces Detected: ' + a, (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

while 1:

    # Algılandıktan sonra, burada yüz olan ve etrafına bir dikdörtgen çizen bir ROI (ilgi bölgesi) oluşturulur.
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        roi_gray = gray[y:y + h, x:x + w]
        # gülümseme kademeli sınıflandırıcının 'DetectMultiscale' işlevi kullanılır.
        roi_img = img[y:y + h, x:x + w]
        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=22, minSize=(25, 25))
        for (sx, sy, sw, sh) in smile:
            # Algılandığında bunun için bir ROI oluşturulur ve etrafına bir dikdörtgen çizilir.
            cv2.rectangle(roi_img, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            # ROI alanı ve gülümsemenin oranı hesaplanır.
            sm_ratio = str(round(sw / sh, 3))  #sm_ratio, gülümserken veya gülerken ağız genişliğinin ve yüksekliğinin oranını ifade eder.
            #Burada uygulanan mantık, bir kişi daha çok gülümsediğinde veya daha çok güldüğünde ağız genişliğinin artması,
            #buna karsılık sm_ratio'nun artmasıdır.
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Smile meter : ' + sm_ratio, (x + (-25), y + h - (-25)), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if float(sm_ratio) > 1.8:
                cv2.putText(img, "Smiling", (x - 160, y + h - 160), cv2.FONT_HERSHEY_SIMPLEX, 2, 100, 2, cv2.LINE_AA)
            if float(sm_ratio) > 1.8:
                smile_ratios.append(float(sm_ratio))
                times.append(datetime.now())
    cv2.imshow('Smile Detector', img)
    #cv2.imwrite('imagessmile.png', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
#veriler csv dosyasına aktarılır.
ds = {'smile_ratios': smile_ratios, 'times': times}
df = pd.DataFrame(ds)
df.to_csv('smile_records_image.csv')

df = pd.read_csv('C:\\Users\\DK\\PycharmProjects\\GoruntuIslemeProje\\meter\\smile_records_image.csv')

fig = px.line(df, x='smile_ratios', y='times', title='Smile Records Image')

fig.show()


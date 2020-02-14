import cv2
import sys
from django.conf import settings
import numpy as np

import datetime


class FaceDection:
    def __init__(self):
        print("computerVision..")
        filename= settings.MEDIA_ROOT + 'haarcascade_frontalface_default.xml'
        self.faceCascade = cv2.CascadeClassifier(filename)
        self.scale= 100

    def get_frame(self):
        cap =cv2.VideoCapture(0) 
        while True:
            _, frame = cap.read()

            frame, frameCrops = self.imgFaceDetection(frame)# face detection

            if frameCrops:
                frameMerge= self.merge(frame, frameCrops[0])

            imgencode=cv2.imencode('.jpg',frameMerge)[1]
            stringData=imgencode.tostring()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
            
        del(cap)

    def imgFaceDetection(self, frame):
        t1= datetime.datetime.now()

        frameCrops=[]
        faces= self.faceBoundingbox(frame)
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            imgFace = frame[y:y+h, x:x+w]
            imgFace10by10= self.resize(imgFace)
            frameCrops.append( imgFace10by10 )

        #frame = np.concatenate((frame, frame), axis=0)

        t2= datetime.datetime.now()
        t12 = (t2 - t1).total_seconds()
        #print("processing one image of face detection: ", t12, 's')

        return frame, frameCrops

    def faceBoundingbox(self, frame): # one image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #faces = self.faceCascade.detectMultiScale(
        f= settings.MEDIA_ROOT + 'haarcascade_frontalface_default.xml'
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        return faces

    def resize(self, img):
        dim = (self.scale, self.scale)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        #print('Resized Dimensions : ',resized.shape)
        return resized

    def merge(self, img1, img2):

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        #create empty martrix (Mat)
        res = np.zeros(shape=(max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        # assign BGR values to concatenate images
        for i in range(res.shape[2]):
            # assign img1 colors
            res[:h1, :w1, i] = np.ones([img1.shape[0], img1.shape[1]]) * img1[:, :, i]
            # assign img2 colors
            res[:h2, w1:w1 + w2, i] = np.ones([img2.shape[0], img2.shape[1]]) * img2[:, :, i]

        output_img = res.astype('uint8')
        
        return output_img

    def record(self ,cap):
        frame_width = int(cap.get(3)) 
        frame_height = int(cap.get(4))
        outf= 'save.avi'
        out = cv2.VideoWriter(outf,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        return out


if __name__== "__main__":
    FaceDection().pipeline()

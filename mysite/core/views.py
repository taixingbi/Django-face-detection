import os
import json

import logging
logger = logging.getLogger(__name__)
from django.http import HttpResponse

from django.http import Http404
from django.http import HttpResponseServerError
from django.core.exceptions import EmptyResultSet

from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, CreateView
from django.http import JsonResponse
from django.utils import timezone

# from django.contrib.staticfiles import finders
import pandas as pd 

 #db
from database.orm import DBRead

#s3
from aws.s3 import s3Bucket
from aws.ses import SES

from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import logout

import time

from django.conf import settings

from ml.faceDetection import FaceDection

from django.shortcuts import render
from django.http import HttpResponse,StreamingHttpResponse, HttpResponseServerError
from django.views.decorators import gzip
import cv2
import time

from django.contrib.sites.models import Site



# views
class Home(TemplateView):
    template_name = 'home.html'

#logout
def logout_view(request):
    logout(request)
    return render(request,'home.html')

#camera
def camera_view(request):
    #camera(request)

    return render(request, 'camera.html', {'BASIC_DOMAIN': settings.BASIC_DOMAIN} )


# def get_frame():
#     camera =cv2.VideoCapture(0) 
#     while True:
#         _, img = camera.read()
#         imgencode=cv2.imencode('.jpg',img)[1]
#         stringData=imgencode.tostring()
#         yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

#     del(camera)
    

dataTime= {
    "time": timezone.localtime(),
}

# -----------------------------------------for api-------------------------------------
class Demo(): 

    @gzip.gzip_page
    def test(request, stream_path="video"):
        frame= FaceDection().get_frame()

        try :
            return StreamingHttpResponse(frame,content_type="multipart/x-mixed-replace;boundary=frame")
        except :
            return "error"

    #/test/s3/
    def s3(request):  
        print("\n\n************************************* s3 test*************************************")

        #bucket='thrivee-dev/audiotranscribe'
        bucket=  'thrivee-dev'

        key= 'audiotranscribe/test.wav'

        fileName= 'media/' + key.split('/')[1]
        print(fileName)
        res= s3Bucket(bucket, key, fileName).loadFile()

        data= {
            "s3": res,
        }
        
        return JsonResponse(data)
    
    #/test/db
    def db(request):  
        print("\n\n************************************* db test*************************************")
        email= DBRead().test() 

        data= {
            "db test": "success",
            "email": email,
        }
        
        return JsonResponse(data)

    def ses(request):  
        print("\n\n************************************* ese test*************************************")

        SES().gmail()

        data= {
            "ses": "ses",
        }
        
        return JsonResponse(data)

    #/api/demo
    def demo(request):  #s3 key
        print("\n\n*************************************face detection  *************************************")

        print(settings.MEDIA_ROOT)

        data= {
            "demo": "demo",
        }
        
        return JsonResponse(data)

        # df= pd.DataFrame(data, index=[0])
        # return HttpResponse( df.to_html() )
    

          

from rest_framework.decorators import api_view, permission_classes, list_route
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework import serializers
from django.views.generic.base import TemplateView
import cv2
import numpy
from withoutKeras.src.Model import Model, DecoderType
from withoutKeras.src.DataLoader import DataLoader, Batch
from withoutKeras.src.SamplePreprocessor import preprocess



import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def rel(*x):
	return os.path.join(BASE_DIR, *x)

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)


class FilePaths:
	"filenames and paths to data"
	fnCharList = rel('withoutKeras/model/charList.txt')
	fnAccuracy = rel('withoutKeras/model/accuracy.txt')
	fnTrain = rel('withoutKeras/data/')
	fnInfer = rel('withoutKeras/data/test3.png')
	fnCorpus = rel('withoutKeras/data/corpus.txt')


def infer(model, img):
	"recognize text in image provided by file path"
	ret, img = cv2.threshold(img, 155, 255, cv2.THRESH_BINARY)
	img = preprocess(img, Model.imgSize)
	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)
	print('Recognized:', '"' + recognized[0] + '"')
	print('Probability:', probability[0])
	return recognized[0]


@api_view(['post'])
def get_string(request):
	print(request.data)
	print(request.FILES['file'])

	model = Model(open(FilePaths.fnCharList).read(), DecoderType.BestPath, mustRestore=True)



	img = cv2.imdecode(numpy.fromstring(request.FILES['file'].read(), numpy.uint8), cv2.IMREAD_GRAYSCALE)
	text=infer(model, img)
	return Response(data={'status':"ok",'text':text}, status=status.HTTP_200_OK)


class imageView(TemplateView):
	template_name = 'app.html'
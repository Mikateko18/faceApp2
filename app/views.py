from fileinput import filename
from PIL import Image
import cv2
from flask import render_template, request
from flask import redirect, url_for
import os
import glob
from cv2 import threshold
import numpy as np
from numpy import linalg as la
from pylab import *
import matplotlib.pyplot as plt 

def faceApp():
    
    if request.method == "POST":
        f = request.files['image']
        filename=  f.filename
        path = os.path.join(training_images_directory,filename)
        f.save(path)
        w = getwidth(path)
        #grayImage= np.zeros_like(f)
        

        return render_template('faceApp.html',fileupload=True,imageName=filename, w=w)


    return render_template('faceApp.html',fileupload=False,imageName="002.jpeg")

training_images_directory = "static/uploads"
images = os.listdir(training_images_directory)

for count, imageName in enumerate(images,1):
      image = plt.imread(os.path.join(training_images_directory,imageName))

def getLocalBinaryPatternImage(grayImage):
  imageLocalBinaryPattern = np.zeros_like(grayImage)
  neighboor = 3
  for imageHeight in range(0,image.shape[0] - neighboor):
    for imageWidth in range(0,image.shape[1] - neighboor):
      images          = grayImage[imageHeight:imageHeight+neighboor,imageWidth:imageWidth+neighboor]
      center       = images[1,1]
      image01        = (images >= center)*1.0
      image01Vector = image01.T.flatten()
      image01Vector = np.delete(image01Vector,4)
      whereImage01Vector = np.where(image01Vector)[0]
      if len(whereImage01Vector) >= 1:
        num = np.sum(2**whereImage01Vector)
      else:
        num = 0
      imageLocalBinaryPattern[imageHeight+1,imageWidth+1] = num
  
  return(imageLocalBinaryPattern)

training_images_directory  = "static/uploads"
images = os.listdir(training_images_directory )
for imageName in images:
  image = plt.imread(os.path.join(training_images_directory ,imageName))
  imageLocalBinaryPattern    = getLocalBinaryPatternImage(image)
  vecimgLBP = imageLocalBinaryPattern.flatten()
  figure = plt.figure(figsize=(22,10))
  axis  = figure.add_subplot(1,3,1)
  axis.imshow(image)
  axis.set_title(imageName)
  axis  = figure.add_subplot(1,3,2)
  axis.imshow(imageLocalBinaryPattern)
  axis.set_title("Local Binary Pattern converted image")
  axis  = figure.add_subplot(1,3,3)
  freq,localBinaryPattern, _ = axis.hist(vecimgLBP,bins=2**8)
  axis.set_ylim(0,40000)
  localBinaryPattern = localBinaryPattern[:-1]
  largeTF = freq > 5000
  
  axis.set_title("Local Binary Pattern histogram")
  plt.savefig( './static/LBPImages/'+ imageName)

  def image_data():
      imagesPaths = [os.path.join('./static/LBPImages', imageList) for imageList in os.listdir('./static/LBPImages')]
      imagesPaths = [i for i in imagesPaths if os.path.isfile(i)]
      imageFaces = []
      imageIds = []
      for path in imagesPaths:
          faceImage = Image.open(path).convert('L')
          imageNumpy = np.array(faceImage, 'uint8')
          faceId = int(os.path.split(path)[1].split('.')[0])
          imageFaces.append(imageNumpy)
          imageIds.append(faceId)
      return imageFaces, np.array(imageIds)

imageFaces, imageIds = image_data()
#print(imageIds)

lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.train(imageFaces, imageIds)
lbph_classifier.write('lbph_classifier.yml')
#print('Created')

lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read('./lbph_classifier.yml')
#print('read')

test_image = 'static/uploads/6.jpg'

image = Image.open(test_image).convert('L')
image_np = np.array(image, 'uint8')
image_np

prediction = lbph_face_classifier.predict(image_np)
#print(prediction)

expected_output = int(os.path.split(test_image)[1].split('.')[0].replace('subject', ''))
#print(expected_output)

    
cv2.putText(image_np, 'Prediction: ' + str(prediction), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, color = (0, 0, 255))
cv2.putText(image_np, 'Expected: ' + str(expected_output), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0,0,255))
cv2.imwrite('./static/predict/{}'.format('6.jpg'),image_np)
print('Saved')


def getwidth(path):
    img = Image.open(path)
    size = img.size # width and height
    aspect = size[0]/size[1] # width / height
    w = 380 * aspect
    return int(w)
        

        
   
    

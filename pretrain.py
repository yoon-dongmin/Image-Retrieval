import os
from torchvision import models
from torchvision import transforms
import torch
import cv2
from PIL import Image
import numpy as np
import urllib 
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np

def classify(image):
    plt.close('all')
    resnet101_pretrained = models.resnet101(pretrained=True)

    resnet101_pretrained.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    input_image = Image.open('robot/'+image)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    # imshow(np.asarray(input_image))
    # plt.show()
    #cv2.imshow(111,input_image)
    # move the input and model to GPU for speed if available 
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        resnet101_pretrained.to('cuda') 

    with torch.no_grad(): #backprop 하지 않겠다
        # Perform inference.  
        output = resnet101_pretrained(input_batch)

    # Download the class file.  #1000개의 class로 구성
    url, filename = ("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json", "imagenet_classes.json")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename) 

    # Load class file. 
    imagenet_class_index = json.load(open('imagenet_classes.json')) 

    # Find out the index where the maximum score in output vector occurs.
    _, index = torch.max(output, 1) #가장 높은 score의 index만 가지고 옴

    # Calculate the softmax value to get the percentage of accruacy. 
    percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100   #가장 높은 score의 정확도 계산
    #print("Accuracy : ", percentage[index[0]].item(), "%")

    # Find the index of class 
    predicted_idx = str(index.item())
    print("Do you mean this", imagenet_class_index[predicted_idx][1]+"?")
    imshow(np.asarray(input_image))
    plt.show()
    
    #text1 = input()
    #plt.cla()
    #plt.close()


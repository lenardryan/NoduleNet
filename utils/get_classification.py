import numpy as np
import os
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torchvision.models import VGG16_Weights
from torchvision import transforms
from collections import OrderedDict
from PIL import Image

model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    
for param in model.parameters():
    param.requires_grad = False

# reduce output to 3
layers_vgg16 = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, 512)),
    ('activation1', nn.ReLU()),
    ('dropout1', nn.Dropout()),
    ('fc2', nn.Linear(512, 256)),
    ('activation2', nn.ReLU()),
    ('dropout2', nn.Dropout()),
    ('fc3', nn.Linear(256, 128)),
    ('activation3', nn.ReLU()),
    ('dropout3', nn.Dropout()),
    ('fc4', nn.Linear(128, 3))
]))
model.classifier = layers_vgg16

# Uncomment these to load a saved model

model.load_state_dict(torch.load('/content/drive/MyDrive/Thesis/NoduleNet/utils/model.ckpt'))
### now you can evaluate it
model.eval()
if torch.cuda.is_available():
  model.cuda()

def get_class(image):

  image = Image.fromarray(np.uint8(image)*255)

  tfms = transforms.Compose([
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))    # standard normalization values needed for vgg16
  ])
  image = tfms(image)

  classes = ['Benign', 'Metastatic', 'Primary']

  with torch.no_grad():
    pred = model(image.cuda().unsqueeze(0))
  
  pred_class = classes[pred.argmax().item()]

  return pred_class
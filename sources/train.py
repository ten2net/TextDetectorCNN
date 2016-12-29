from models import *
from data import *
from keras.optimizers import *
model_name='fcn'
weights_path=''
optimizer=adadelta()
if model_name=='fcn':
    model=FCN()
    model.train(optimizer,weights_path)
if model_name=='segnet':
    model=SegNet()
    model.train(optimizer,weights_path)
if model_name=='exsegnet':
    model=ExSegNet()
    model.train(optimizer,weights_path)
if model_name=='smsegnet':
    model=SmSegNet()
    model.train(optimizer,weights_path)
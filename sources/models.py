from data import *
from keras import backend
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils.visualize_util import plot
from theano import tensor
class UnPooling2D(UpSampling2D):
    input_ndim=4
    def __init__(self,pool2d_layer,*args,**kwargs):
        self._pool2d_layer=pool2d_layer
        super().__init__(*args,**kwargs)
    def get_output(self,train=False):
        X=self.get_input(train)
        if self.dim_ordering=='th':
            output=backend.repeat_elements(backend,self.size[0],axis=2)
            output=backend.repeat_elements(output,self.size[1],axis=3)
        else:
            output=backend.repeat_elements(backend,self.size[0],axis=1)
            output=backend.repeat_elements(output,self.size[1],axis=2)
        f=tensor.grad(tensor.sum(
            self._pool2d_layer.get_output(train)),
            wrt=self._pool2d_layer.get_input(train))*output
        return f
class SmSegNet:
    def __init__(self,weights_file=''):
        p=[]
        model=Sequential()
        model.add(Convolution2D(32,3,3,border_mode='same',input_shape=(3,224,224)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(32,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        p+=[MaxPooling2D((2,2),strides=(2,2))]
        model.add(p[-1])
        model.add(Convolution2D(128,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(128,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        p+=[MaxPooling2D((2,2),strides=(2,2))]
        model.add(p[-1])
        model.add(Convolution2D(256,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(256,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(256,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        p+=[MaxPooling2D((2,2),strides=(2,2))]
        model.add(p[-1])
        model.add(Convolution2D(256,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(256,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        p+=[MaxPooling2D((2,2),strides=(2,2))]
        model.add(p[-1])
        model.add(Convolution2D(256,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(256,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(256,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Deconvolution2D(256,3,3,output_shape=(None,256,16,16)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(UnPooling2D(p.pop(-1),size=(2,2)))
        model.add(Deconvolution2D(256,3,3,output_shape=(None,256,30,30)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(UnPooling2D(p.pop(-1),size=(2,2)))
        model.add(Deconvolution2D(256,3,3,output_shape=(None,256,58,58)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(UnPooling2D(p.pop(-1),size=(2,2)))
        model.add(Deconvolution2D(128,3,3,output_shape=(None,128,114,114)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(UnPooling2D(p.pop(-1),size=(2,2)))
        model.add(Deconvolution2D(32,3,3,output_shape=(None,32,226,226)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(1,3,3,output_shape=(None,1,226,226)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('sigmoid'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Reshape((224,224)))
        self.model=model
        self.load(weights_file)
        self.plot()
    def summary(self):
        self.model.summary()
    def plot(self):
        plot(self.model,'..\\visuals\\graph_smsegnet.png')
    def load(self,weights_file):
        if weights_file:
            self.model.load_weights("..\\weights\\"+weights_file+".hdf5")
    def train(self,optimizer,weights_file):
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"])
        if weights_file:
            self.model.load_weights("..\\weights\\"+weights_file+".hdf5")
        checkpointer=ModelCheckpoint(
            filepath="..\\weights\\weights_best.hdf5",
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            save_weights_only=True)
        self.model.fit_generator(
            generator('train',8),
            samples_per_epoch=1024,
            nb_epoch=65536,
            callbacks=[checkpointer],
            validation_data=generator('validate',8),
            nb_val_samples=256)
    def predict(self,x):
        return self.model.predict_on_batch(x)
class ExSegNet:
    def __init__(self,weights_file=''):
        p=[]
        model=Sequential()
        model.add(Convolution2D(128,3,3,border_mode='same',input_shape=(3,224,224)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(128,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        p+=[MaxPooling2D((2,2),strides=(2,2))]
        model.add(p[-1])
        model.add(Convolution2D(256,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(256,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        p+=[MaxPooling2D((2,2),strides=(2,2))]
        model.add(p[-1])
        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        p+=[MaxPooling2D((2,2),strides=(2,2))]
        model.add(p[-1])
        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        p+=[MaxPooling2D((2,2),strides=(2,2))]
        model.add(p[-1])
        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        p+=[MaxPooling2D((2,2),strides=(2,2))]
        model.add(p[-1])
        model.add(UnPooling2D(p.pop(-1),size=(2,2)))
        model.add(Deconvolution2D(512,3,3,output_shape=(None,512,16,16)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(512,3,3,output_shape=(None,512,16,16)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(512,3,3,output_shape=(None,512,16,16)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(UnPooling2D(p.pop(-1),size=(2,2)))
        model.add(Deconvolution2D(512,3,3,output_shape=(None,512,30,30)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(512,3,3,output_shape=(None,512,30,30)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(512,3,3,output_shape=(None,512,30,30)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(UnPooling2D(p.pop(-1),size=(2,2)))
        model.add(Deconvolution2D(512,3,3,output_shape=(None,512,58,58)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(512,3,3,output_shape=(None,512,58,58)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(512,3,3,output_shape=(None,512,58,58)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(UnPooling2D(p.pop(-1),size=(2,2)))
        model.add(Deconvolution2D(256,3,3,output_shape=(None,256,114,114)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(256,3,3,output_shape=(None,256,114,114)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(UnPooling2D(p.pop(-1),size=(2,2)))
        model.add(Deconvolution2D(128,3,3,output_shape=(None,128,226,226)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(128,3,3,output_shape=(None,128,226,226)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(1,3,3,output_shape=(None,1,226,226)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('sigmoid'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Reshape((224,224)))
        self.model=model
        self.load(weights_file)
        self.plot()
    def summary(self):
        self.model.summary()
    def plot(self):
        plot(self.model,'..\\visuals\\graph_exsegnet.png')
    def load(self,weights_file):
        if weights_file:
            self.model.load_weights("..\\weights\\"+weights_file+".hdf5")
    def train(self,optimizer,weights_file):
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"])
        if weights_file:
            self.model.load_weights("..\\weights\\"+weights_file+".hdf5")
        checkpointer=ModelCheckpoint(
            filepath="..\\weights\\weights_best.hdf5",
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            save_weights_only=True)
        self.model.fit_generator(
            generator('train',2),
            samples_per_epoch=1024,
            nb_epoch=65536,
            callbacks=[checkpointer],
            validation_data=generator('validate',2),
            nb_val_samples=256)
    def predict(self,x):
        return self.model.predict_on_batch(x)
class SegNet:
    def __init__(self,weights_file=''):
        p=[]
        model=Sequential()
        model.add(Convolution2D(64,3,3,border_mode='same',input_shape=(3,224,224)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(64,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        p+=[MaxPooling2D((2,2),strides=(2,2))]
        model.add(p[-1])
        model.add(Convolution2D(128,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(128,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        p+=[MaxPooling2D((2,2),strides=(2,2))]
        model.add(p[-1])
        model.add(Convolution2D(256,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(256,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(256,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        p+=[MaxPooling2D((2,2),strides=(2,2))]
        model.add(p[-1])
        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        p+=[MaxPooling2D((2,2),strides=(2,2))]
        model.add(p[-1])
        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        p+=[MaxPooling2D((2,2),strides=(2,2))]
        model.add(p[-1])
        model.add(UnPooling2D(p.pop(-1),size=(2,2)))
        model.add(Deconvolution2D(512,3,3,output_shape=(None,512,16,16)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(512,3,3,output_shape=(None,512,16,16)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(512,3,3,output_shape=(None,512,16,16)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(UnPooling2D(p.pop(-1),size=(2,2)))
        model.add(Deconvolution2D(512,3,3,output_shape=(None,512,30,30)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(512,3,3,output_shape=(None,512,30,30)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(512,3,3,output_shape=(None,512,30,30)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(UnPooling2D(p.pop(-1),size=(2,2)))
        model.add(Deconvolution2D(256,3,3,output_shape=(None,256,58,58)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(256,3,3,output_shape=(None,256,58,58)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(256,3,3,output_shape=(None,256,58,58)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(UnPooling2D(p.pop(-1),size=(2,2)))
        model.add(Deconvolution2D(128,3,3,output_shape=(None,128,114,114)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(128,3,3,output_shape=(None,128,114,114)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(UnPooling2D(p.pop(-1),size=(2,2)))
        model.add(Deconvolution2D(64,3,3,output_shape=(None,64,226,226)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(64,3,3,output_shape=(None,64,226,226)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Deconvolution2D(1,3,3,output_shape=(None,1,226,226)))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('sigmoid'))
        model.add(Cropping2D(cropping=((1,1),(1,1))))
        model.add(Reshape((224,224)))
        self.model=model
        self.load(weights_file)
        self.plot()
    def summary(self):
        self.model.summary()
    def plot(self):
        plot(self.model,'..\\visuals\\graph_segnet.png')
    def load(self,weights_file):
        if weights_file:
            self.model.load_weights("..\\weights\\"+weights_file+".hdf5")
    def train(self,optimizer,weights_file):
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"])
        if weights_file:
            self.model.load_weights("..\\weights\\"+weights_file+".hdf5")
        checkpointer=ModelCheckpoint(
            filepath="..\\weights\\weights_best.hdf5",
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            save_weights_only=True)
        self.model.fit_generator(
            generator('train',8),
            samples_per_epoch=1024,
            nb_epoch=65536,
            callbacks=[checkpointer],
            validation_data=generator('validate',8),
            nb_val_samples=288*12)
    def predict(self,x):
        return self.model.predict_on_batch(x)
class FCN:
    def __init__(self,weights_file=''):
        x=Input(shape=(3,224,224))
        y=Convolution2D(64,3,3,border_mode='same')(x)
        y=BatchNormalization(axis=1)(y)
        y=Activation('relu')(y)
        y=Convolution2D(64,3,3,border_mode='same')(y)
        y=BatchNormalization(axis=1)(y)
        y=Activation('relu')(y)
        y=MaxPooling2D((2,2),strides=(2,2))(y)
        y=Convolution2D(128,3,3,border_mode='same')(y)
        y=BatchNormalization(axis=1)(y)
        y=Activation('relu')(y)
        y=Convolution2D(128,3,3,border_mode='same')(y)
        y=BatchNormalization(axis=1)(y)
        y=Activation('relu')(y)
        y=MaxPooling2D((2,2),strides=(2,2))(y)
        y=Convolution2D(256,3,3,border_mode='same')(y)
        y=BatchNormalization(axis=1)(y)
        y=Activation('relu')(y)
        y=Convolution2D(256,3,3,border_mode='same')(y)
        y=BatchNormalization(axis=1)(y)
        y=Activation('relu')(y)
        y=Convolution2D(256,3,3,border_mode='same')(y)
        y=BatchNormalization(axis=1)(y)
        y=Activation('relu')(y)
        y=MaxPooling2D((2,2),strides=(2,2))(y)
        p3=Convolution2D(1,1,1)(y)
        p3=BatchNormalization(axis=1)(p3)
        p3=Activation('relu')(p3)
        y=Convolution2D(512,3,3,border_mode='same')(y)
        y=BatchNormalization(axis=1)(y)
        y=Activation('relu')(y)
        y=Convolution2D(512,3,3,border_mode='same')(y)
        y=BatchNormalization(axis=1)(y)
        y=Activation('relu')(y)
        y=Convolution2D(512,3,3,border_mode='same')(y)
        y=BatchNormalization(axis=1)(y)
        y=Activation('relu')(y)
        y=MaxPooling2D((2,2),strides=(2,2))(y)
        p4=Convolution2D(1,1,1)(y)
        p4=BatchNormalization(axis=1)(p4)
        p4=Activation(activation='relu')(p4)
        p4=Deconvolution2D(1,4,4,output_shape=(None,1,30,30),subsample=(2,2))(p4)
        p4=BatchNormalization(axis=1)(p4)
        p4=Activation('relu')(p4)
        p4=Cropping2D(cropping=((1,1),(1,1)))(p4)
        y=Convolution2D(512,3,3,border_mode='same')(y)
        y=BatchNormalization(axis=1)(y)
        y=Activation('relu')(y)
        y=Convolution2D(512,3,3,border_mode='same')(y)
        y=BatchNormalization(axis=1)(y)
        y=Activation('relu')(y)
        y=Convolution2D(512,3,3,border_mode='same')(y)
        y=BatchNormalization(axis=1)(y)
        y=Activation('relu')(y)
        y=MaxPooling2D((2,2),strides=(2,2))(y)
        p5=Convolution2D(1,1,1,activation='relu')(y)
        p5=BatchNormalization(axis=1)(p5)
        p5=Activation('relu')(p5)
        p5=Deconvolution2D(1,8,8,output_shape=(None,1,32,32),subsample=(4,4))(p5)
        p5=BatchNormalization(axis=1)(p5)
        p5=Activation('relu')(p5)
        p5=Cropping2D(cropping=((2,2),(2,2)))(p5)
        y=merge([p3,p4,p5])
        y=Deconvolution2D(1,16,16,output_shape=(None,1,232,232),subsample=(8,8))(y)
        y=Cropping2D(cropping=((4,4),(4,4)))(y)
        y=Reshape((224,224))(y)
        y=BatchNormalization(axis=1)(y)
        y=Activation('sigmoid')(y)
        model=Model(x,y)
        self.model=model
        self.load(weights_file)
        self.plot()
    def summary(self):
        self.model.summary()
    def plot(self):
        plot(self.model,'..\\visuals\\graph_fcn.png')
    def load(self,weights_file):
        if weights_file:
            self.model.load_weights("..\\weights\\"+weights_file+".hdf5")
    def train(self,optimizer,weights_file):
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"])
        if weights_file:
            self.model.load_weights("..\\weights\\"+weights_file+".hdf5")
        checkpointer=ModelCheckpoint(
            filepath="..\\weights\\weights_best.hdf5",
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            save_weights_only=True)
        self.model.fit_generator(
            generator('train',16),
            samples_per_epoch=1024,
            nb_epoch=65536,
            callbacks=[checkpointer],
            validation_data=generator('validate',16),
            nb_val_samples=144)#'''288*12'''
    def predict(self,x):
        return self.model.predict_on_batch(x)

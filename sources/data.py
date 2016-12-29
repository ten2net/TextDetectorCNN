import cv2
import math
import numpy
import os
import random
def rotate_image(image,angle):
    image_center=tuple(numpy.array(image.shape)[:2]/2)
    rot_mat=cv2.getRotationMatrix2D(image_center,angle,1.0)
    result=cv2.warpAffine(image,rot_mat,image.shape[:2],flags=cv2.INTER_LINEAR)
    return result
def transform_image(image):
    image=image.astype('float32')
    image=image*2/255-numpy.ones(image.shape)
    return image.transpose((2,0,1))
def generator(stage,batch_size):
    path=os.path.join(os.getenv('DATA'),'TD23(512,line)',stage)
    names=[i for i in list(set(map(lambda x: x.split('.')[0],os.listdir(path)))) if '_gt' not in i]
    random.shuffle(names)
    i=0
    que=[]
    while 1:
        batch=(numpy.zeros((batch_size,3,224,224)),numpy.zeros((batch_size,224,224)))
        for j in range(batch_size):
            if not que:
                for k in range(288):
                    name=names[i]
                    a=random.uniform(0,360)
                    x=cv2.imread(path + '\\' + name + '.jpg')
                    x=rotate_image(x,a)
                    y=cv2.imread(path + '\\' + name + '_gt.jpg',cv2.IMREAD_GRAYSCALE)
                    y=rotate_image(y,a)
                    d=random.randint(224,324)
                    x=cv2.resize(x,(d,d))
                    y=cv2.resize(y,(d,d))
                    tx=numpy.ones(x.shape)*255-x
                    x=transform_image(x)
                    tx = transform_image(tx)
                    y=y.astype('float32') / 255
                    que.append((x[:,:224,:224],y[:224,:224],name))
                    que.append((x[:,-224:,-224 :],y[ -224:,-224 :],name))
                    que.append((x[:,:224,-224 :],y[:224,-224 :],name))
                    que.append((x[:,-224:,:224],y[ -224:,:224],name))
                    x=x[:,::-1,:]
                    tx = tx[:, ::-1, :]
                    y=y[::-1,:]
                    que.append((x[:,:224,:224],y[:224,:224],name))
                    que.append((x[:,-224:,-224:],y[-224:,-224:],name))
                    que.append((x[:,:224,-224:],y[:224,-224:],name))
                    que.append((x[:,-224:,:224],y[-224:,:224],name))
                    x=tx
                    que.append((x[:,:224,:224],y[:224,:224],name))
                    que.append((x[:,-224:,-224:],y[-224:,-224:],name))
                    que.append((x[:,:224,-224:],y[:224,-224:],name))
                    que.append((x[:,-224:,:224],y[-224:,:224],name))
                    if 'IMG' in name and stage=='train':
                        a = random.uniform(0, 360)
                        x = cv2.imread(path + '\\' + name + '.jpg')
                        x = rotate_image(x, a)
                        y = cv2.imread(path + '\\' + name + '_gt.jpg', cv2.IMREAD_GRAYSCALE)
                        y = rotate_image(y, a)
                        d = 224*2
                        x = cv2.resize(x, (d, d))
                        y = cv2.resize(y, (d, d))
                        tx = numpy.ones(x.shape) * 255 - x
                        x = transform_image(x)
                        tx = transform_image(tx)
                        y = y.astype('float32') / 255
                        que.append((x[:, :224, :224], y[:224, :224], name))
                        que.append((x[:, -224:, -224:], y[-224:, -224:], name))
                        que.append((x[:, :224, -224:], y[:224, -224:], name))
                        que.append((x[:, -224:, :224], y[-224:, :224], name))
                        x = x[:, ::-1, :]
                        tx = tx[:, ::-1, :]
                        y = y[::-1, :]
                        que.append((x[:, :224, :224], y[:224, :224], name))
                        que.append((x[:, -224:, -224:], y[-224:, -224:], name))
                        que.append((x[:, :224, -224:], y[:224, -224:], name))
                        que.append((x[:, -224:, :224], y[-224:, :224], name))
                        x = tx
                        que.append((x[:, :224, :224], y[:224, :224], name))
                        que.append((x[:, -224:, -224:], y[-224:, -224:], name))
                        que.append((x[:, :224, -224:], y[:224, -224:], name))
                        que.append((x[:, -224:, :224], y[-224:, :224], name))
                    i=(i+1)%len(names)
                random.shuffle(que)
            batch[0][j],batch[1][j]=que.pop(0)[:2]
        yield batch
if __name__=='__main__':
    g=generator('train',8)
    tt=0
    for (x,y) in g:
        for i in range(8):
            tx=x[i].transpose([1,2,0])
            tx+=numpy.ones(tx.shape)
            tx/=2
            tx*=255
            tx=tx.astype('uint8')
            ty=(y[i]*255).astype('uint8')
            cv2.imwrite('..\\visual\\data\\'+'%04d'%tt+'.jpg',tx)
            cv2.imwrite('..\\visual\\data\\'+'%04d'%tt + 'gt.jpg',ty)
            tt+=1
        if tt>400:
            break
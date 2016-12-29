import os
import time
import cv2
import detect
work_dir=time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
if not os.path.isdir(os.path.join('..\\results',work_dir)):
    os.mkdir(os.path.join('..\\results',work_dir))
test_subject='get_com_all_td23'
if test_subject=='mser':
    image_list=list(set(map(lambda x: x.split('.')[0],os.listdir('..\\tests\\MSRA-TD500(validate)'))))
    image_list.sort()
    for image_name in image_list:
        image=cv2.imread('..\\tests\\MSRA-TD500(validate)\\'+image_name+'.jpg')
        result=detect.mser_gray(image,debug=1)[1]
        cv2.imwrite('..\\results\\'+work_dir+'\\'+image_name+'_grey.jpg',result)
        result = detect.mser_gray_refined(image, debug=1)[1]
        cv2.imwrite('..\\results\\' + work_dir + '\\' + image_name + '_grey_refined.jpg', result)
        result = detect.mser_color(image, debug=1)[1]
        cv2.imwrite('..\\results\\' + work_dir + '\\' + image_name + '_color.jpg', result)
if test_subject=='get_map':
    image_list = list(set(map(lambda x: x.split('.')[0], os.listdir('..\\tests\\MSRA-TD500(validate)'))))
    image_list.sort()
    for image_name in image_list:
        image = cv2.imread('..\\tests\\MSRA-TD500(validate)\\' + image_name + '.jpg')
        image=cv2.resize(image,(224*4,224*4))
        cv2.imwrite('..\\results\\' + work_dir + '\\' + image_name + '_org.jpg', image)
        result=detect.get_map(image)
        cv2.imwrite('..\\results\\' + work_dir + '\\' + image_name + '_map.jpg', image)
if test_subject=='get_com':
    image_list = list(set(map(lambda x: x.split('.')[0], os.listdir('..\\tests\\MSRA-TD500(validate)'))))
    image_list.sort()
    for image_name in image_list:
        image = cv2.imread('..\\tests\\MSRA-TD500(validate)\\' + image_name + '.jpg')
        image=cv2.resize(image,(224*4,224*4))
        result = detect.get_components(image,debug=1)
        cv2.imwrite('..\\results\\' + work_dir + '\\' + image_name + '_0.jpg', result[0])
        cv2.imwrite('..\\results\\' + work_dir + '\\' + image_name + '_1.jpg', result[1])
        cv2.imwrite('..\\results\\' + work_dir + '\\' + image_name + '_2.jpg', result[2])
if test_subject=='get_com_all':
    image_list = list(set(map(lambda x: x.split('.')[0], os.listdir('..\\tests\\MSRA-TD500(validate)'))))
    image_list.sort()
    flag=0
    for image_name in image_list:
        print(image_name)
        #if image_name!='IMG_0452':
           #continue
        #if "489" not in image_name:
        #    continue
        image = cv2.imread('..\\tests\\MSRA-TD500(validate)\\' + image_name + '.jpg')
        #image = cv2.resize(image, (224 * 4, 224 * 4))
        result = detect.get_components_all_sizes(image,debug=1)
        for i in range(len(result)):
            cv2.imwrite('..\\results\\' + work_dir + '\\' + image_name + '_%d.jpg'%i, result[i])
if test_subject=='get_com_all_test':
    image_list = list(set(map(lambda x: x.split('.')[0], os.listdir('..\\tests\\MSRA-TD500(test)'))))
    image_list.sort()
    flag=0
    for image_name in image_list:
        #if image_name!='IMG_0479':
        #    continue
        #if "489" not in image_name:
        #    continue
        image = cv2.imread('..\\tests\\MSRA-TD500(test)\\' + image_name + '.jpg')
        #image = cv2.resize(image, (224 * 4, 224 * 4))
        result = detect.get_components_all_sizes(image, debug=1)
        for i in range(len(result)):
            cv2.imwrite('..\\results\\' + work_dir + '\\' + image_name + '_%d.jpg' % i, result[i])
if test_subject=='get_com_all_td23':
    image_list = list(set(map(lambda x: x.split('.')[0], os.listdir('..\\tests\\TD23'))))
    image_list.sort()
    for image_name in image_list:
        if image_name!='IMG_4251':
            continue
        image = cv2.imread('..\\tests\\TD23\\' + image_name + '.jpg')
        #image = cv2.resize(image, (224 * 4, 224 * 4))
        result = detect.get_components_all_sizes(image,debug=1)
        for i in range(len(result)):
            cv2.imwrite('..\\results\\' + work_dir + '\\' + image_name + '_%d.jpg' % i, result[i])
if test_subject=='get_com_all_td233':
    image_list = list(set(map(lambda x: x.split('.')[0], os.listdir('..\\tests\\TD233'))))
    image_list.sort()
    for image_name in image_list:
        #if image_name!='IMG_0479':
        #    continue
        image = cv2.imread('..\\tests\\TD233\\' + image_name + '.png')
        #image = cv2.resize(image, (224 * 4, 224 * 4))
        result = detect.get_components_all_sizes(image,debug=1)
        cv2.imwrite('..\\results\\' + work_dir + '\\' + image_name + '_0.jpg', result[0])
        cv2.imwrite('..\\results\\' + work_dir + '\\' + image_name + '_1.jpg', result[1])
if test_subject=='test2':
    image_list = list(set(map(lambda x: x.split('.')[0], os.listdir('..\\tests\\MSRA-TD500(test)'))))
    image_list.sort()
    for image_name in image_list:
        if image_name!='IMG_2090':
            continue
        image = cv2.imread('..\\tests\\MSRA-TD500(test)\\' + image_name + '.jpg')
        image=cv2.resize(image,(224,224))
        import data
        x = data.transform_image(image)
        x = x.reshape((1,) + x.shape)
        import models
        model = models.SegNet('weights_segnet')
        y = (model.predict(x)[0] * 255).astype('uint8')
        cv2.imwrite('..\\results\\' + work_dir + '\\' + image_name + '_0.jpg', y)

if test_subject=='test1':
    image_list = list(set(map(lambda x: x.split('.')[0], os.listdir('..\\tests\\MSRA-TD500(test)'))))
    image_list.sort()
    flag=0
    tp_all=0
    e_all=0
    to_all=0
    tpd_all=0
    for image_name in image_list:
        #if image_name!='IMG_0479':
        #    continue
        #if "489" not in image_name:
        #    continue
        image = cv2.imread('..\\tests\\MSRA-TD500(test)\\' + image_name + '.jpg')
        #image = cv2.resize(image, (224 * 4, 224 * 4))
        result = detect.get_components_all_sizes(image)
        cv2.imwrite('..\\results\\' + work_dir + '\\' + image_name + '.jpg', result[-1])
        E=[]
        T=[]
        import math
        import numpy
        def vector_len(v):
            return numpy.sum(v ** 2) ** 0.5
        def rotate(cx, cy, x, y, a):
            x, y = x - cx, y - cy
            nx = x * math.cos(a) - y * math.sin(a)
            ny = x * math.sin(a) + y * math.cos(a)
            return [nx + cx, ny + cy]
        for gt in open(os.path.join('..\\tests\\MSRA-TD500(test)', image_name+ '.gt')).readlines():
            i, d, x, y, w, h, a = map(float, gt.split())
            cx, cy = x + w / 2, y + h / 2
            pts = []
            pts.append( x)
            pts.append(x+w)
            pts.append(y)
            pts.append(y+h)
            pts.append(a)
            pts.append(d)
            T.append(pts)
        for r in result[0]:
            v1=r[0]-r[1]
            v2=r[1]-r[2]
            if v1[0]<0:
                v1*=-1
            if v2[0]<0:
                v2*=-1
            if abs(v1[1])<abs(v2[1]):
                v=v1
            else:
                v=v2
            a=math.atan(v[1]/v[0])
            c=(r[0]+r[1]+r[2]+[3])/4
            maxx=-1
            minx=50000
            maxy=-1
            miny=50000
            for p in r:
                tmp=rotate(c[0],c[1],p[0],p[1],-a)
                maxx=max(maxx,tmp[0])
                minx=min(minx,tmp[0])
                maxy = max(maxy, tmp[1])
                miny = min(miny, tmp[1])
            pts = []
            pts.append(minx)
            pts.append(maxx)
            pts.append(miny)
            pts.append(maxy)
            pts.append(a)
            E.append(pts)
        def acc(t):
            for e in E:
                if abs(t[4]-e[4])>=math.pi/8:
                    continue
                xl=max(t[0],e[0])
                xr=min(t[1],e[1])
                yl=max(t[2],e[2])
                yr=min(t[3],e[3])
                dx=max(0,xr-xl)
                dy=max(0,yr-yl)
                at=(t[1]-t[0])*(t[3]-t[2])
                ae=(e[1]-e[0])*(e[3]-e[2])
                ai=dx*dy
                ra=ai/(at+ae-ai)
                if ra>0.5:
                    return True
            return False
        To=[i for i in T if i[5]==0]
        TP=[i for i in T if acc(i)]
        TPd=[i for i in T if i[5]==1 and acc(i)]
        print('E%f T%f To%f TP%f TPd %f'%(len(E),len(T),len(To),len(TP),len(TPd)))
        precision=len(TP)/len(E)
        recall=len(TP)/(len(To)+len(TPd))
        if recall==0 or precision==0:
            f=0
        else:
            f=2/(1/recall+1/precision)
        print('precision %f recall %f f %f'%(precision,recall,f))
        tp_all+=len(TP)
        e_all+=len(E)
        to_all+=len(To)
        tpd_all+=len(TPd)
        precision = tp_all / e_all
        recall = tp_all / (to_all + tpd_all)
        if recall==0 or precision==0:
            f=0
        else:
            f=2/(1/recall+1/precision)
        print('overall precision %f recall %f f %f'%(precision,recall,f))
        pass
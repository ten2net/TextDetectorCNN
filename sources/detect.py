import os
import cv2
import time
import numpy
import data
import models
import random
import math
import matplotlib.path as mplPath
from shapely.geometry import Polygon
model = models.SegNet('weights_segnet')
def mser_gray(image, debug=0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    regions = mser.detectRegions(gray, None)
    rects = [numpy.int0(cv2.boxPoints(cv2.minAreaRect(p.reshape(-1, 1, 2)))) for p in regions]
    if debug:
        image = image.copy()
        cv2.polylines(image, rects, 1, (255, 0, 0))
        return (rects, image)
    return rects
def mser_gray_refined(image, debug=0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create(_max_variation=256)
    regions = mser.detectRegions(gray, None)
    rects = [numpy.int0(cv2.boxPoints(cv2.minAreaRect(p.reshape(-1, 1, 2)))) for p in regions]
    if debug:
        image = image.copy()
        cv2.polylines(image, rects, 1, (255, 0, 0))
        return (rects, image)
    return rects
def mser_color(image, debug=0):
    mser = cv2.MSER_create()
    regions = mser.detectRegions(image, None)
    rects = [numpy.int0(cv2.boxPoints(cv2.minAreaRect(p.reshape(-1, 1, 2)))) for p in regions]
    if debug:
        image = image.copy()
        cv2.polylines(image, rects, 1, (255, 0, 0))
        return (rects, image)
    return rects
def get_map(image, debug=0):
    res = numpy.zeros(image.shape[:-1])
    cnt = numpy.zeros(image.shape[:-1])
    n = image.shape[0]
    for i in range(0, n - 112, 112):
        for j in range(0, n - 112, 112):
            x = data.transform_image(image[i:i + 224, j:j + 224])
            x = x.reshape((1,) + x.shape)
            y = (model.predict(x)[0] * 255)
            res[i:i + 224, j:j + 224] += y
            cnt[i:i + 224, j:j + 224] += 1
    res = (res // cnt).astype('uint8')
    return res
def get_map_fast(image, debug=0):
    print('get_map_fast in')
    from time import clock
    tm = clock()
    res = numpy.zeros(image.shape[:-1])
    cnt = numpy.zeros(image.shape[:-1])
    n = image.shape[0]
    que = []
    for i in range(0, n, 224):
        for j in range(0, n, 224):
            if numpy.max(image[i:i + 224, j:j + 224]) == 0:
                continue
            x = data.transform_image(image[i:i + 224, j:j + 224])
            x = x.reshape((1,) + x.shape)
            que.append((x, i, j))
            # y = (model.predict(x)[0] * 255)
            # res[i:i + 224, j:j + 224] += y
            # cnt[i:i + 224, j:j + 224] += 1
    print('do queue! %d' % len(que))
    ii = 0
    while ii < len(que):
        d = 8
        while d + ii > len(que):
            d //= 2
        x = numpy.zeros((d, 3, 224, 224))
        y = numpy.zeros((d, 224, 224))
        for jj in range(d):
            x[jj] = que[ii + jj][0]
        y = (model.predict(x) * 255)
        for jj in range(d):
            i = que[ii + jj][1]
            j = que[ii + jj][2]
            res[i:i + 224, j:j + 224] += y[jj]
            cnt[i:i + 224, j:j + 224] += 1
        ii += d
    print('do queue done!')
    for i in range(0, n - 112, 112):
        for j in range(0, n - 112, 112):
            if i % 224 == 0 and j % 224 == 0:
                continue
            if numpy.max(res[i:i + 224, j:j + 224]) < 255 * 0.5:
                continue
            x = data.transform_image(image[i:i + 224, j:j + 224])
            x = x.reshape((1,) + x.shape)
            y = (model.predict(x)[0] * 255)
            res[i:i + 224, j:j + 224] += y
            cnt[i:i + 224, j:j + 224] += 1
    res = (res // cnt).astype('uint8')
    print('get_map_fast out %f s\n' % (clock() - tm))
    return res
def remove_duplicated_rects(rects_can):
    # return rects_can
    rects = []
    for rect in rects_can:
        from shapely.geometry import Polygon
        def is_overlap(r1, r2):
            p1 = Polygon(r1.tolist())
            p2 = Polygon(r2.tolist())
            ai = p1.intersection(p2).area
            a1 = p1.area
            a2 = p2.area
            if ai > 0.5 * a1 and ai > 0.5 * a2:
                return True

        fail = 0
        for i in rects:
            if is_overlap(rect, i):
                fail = 1
                break
        if not fail:
            rects.append(rect)
    return rects
def remove_duplicated_rects_fast(rects_can,uuu=0):
    #print("remove_duplicated_rects_fast in")
    def get_area(r):
        A = r[0]
        B = r[1]
        C = r[2]
        D = r[3]
        AD = D - A
        AB = B - A
        LAD = numpy.sum(AD ** 2) ** 0.5
        LAB = numpy.sum(AB ** 2) ** 0.5
        return LAD * LAB
    # return rects_can
    from time import clock
    tm = clock()
    #print("rects number:%d" % len(rects_can))
    rects_can = [(i, get_area(i)) for i in rects_can]
    rects_can.sort(key=lambda x: x[1], reverse=True)
    rects_can = [i[0] for i in rects_can]
    compare_cnt = 0
    rects = []
    for rect in rects_can:
        def get_circle(r):
            A = r[0]
            B = r[1]
            C = r[2]
            D = r[3]
            AD = D - A
            AB = B - A
            LAD = numpy.sum(AD ** 2) ** 0.5
            LAB = numpy.sum(AB ** 2) ** 0.5
            return (A + B + C + D) / 4, max(LAB, LAD) * 1.4142135623731
        def precheck(r1, r2):
            c1, l1 = get_circle(r1)
            c2, l2 = get_circle(r2)
            d = numpy.sum((c1 - c2) ** 2) ** 0.5
            if d > l1 + l2:
                return False
            return True
        def is_overlap(r1, r2):
            if not precheck(r1, r2):
                return False
            p1 = Polygon(r1.tolist())
            p2 = Polygon(r2.tolist())
            ai = p1.intersection(p2).area
            a1 = p1.area
            a2 = p2.area
            if ai > 0.75* a1 and ai > 0.75 * a2:
                return 100000000
            if ai > 0.75 * a1:
                return 1
            return 0
        fail = 0
        for i in range(len(rects))[::-1]:
            compare_cnt += 1
            fail += is_overlap(rect, rects[i])
            if fail > 1:
                break
        if not (fail > 1):
            rects.append(rect)
    #print("compare number:%d" % compare_cnt)
    #print("remain rects:%d" % len(rects))
    #print('remove_duplicated_rects_fast out %f s\n' % (clock() - tm))
    rects_can=rects
    rects=[]
    for rect in rects_can:
        def get_circle(r):
            A = r[0]
            B = r[1]
            C = r[2]
            D = r[3]
            AD = D - A
            AB = B - A
            LAD = numpy.sum(AD ** 2) ** 0.5
            LAB = numpy.sum(AB ** 2) ** 0.5
            return (A + B + C + D) / 4, max(LAB, LAD) * 1.4142135623731
        def precheck(r1, r2):
            c1, l1 = get_circle(r1)
            c2, l2 = get_circle(r2)
            d = numpy.sum((c1 - c2) ** 2) ** 0.5
            if d > l1 + l2:
                return False
            return True
        def is_overlap(r1, r2):
            if not precheck(r1, r2):
                return False
            p1 = Polygon(r1.tolist())
            p2 = Polygon(r2.tolist())
            ai = p1.intersection(p2).area
            a1 = p1.area
            a2 = p2.area
            if ai > 0.75 * a1:
                return ai
            #if ai > 0.5 * a1:
                #return 2
            return 0
        for i in range(len(rects))[::-1]:
            tmp = is_overlap(rect, rects[i][0])
            rects[i][1]+=tmp
        rects.append([rect,0])
    rects_can = rects
    rects = []
    for rect in rects_can:
        if 0.75*get_area(rect[0])>rect[1]:
            rects.append(rect[0])
    rects_can = rects
    rects = []
    for rect in rects_can:
        def get_circle(r):
            A = r[0]
            B = r[1]
            C = r[2]
            D = r[3]
            AD = D - A
            AB = B - A
            LAD = numpy.sum(AD ** 2) ** 0.5
            LAB = numpy.sum(AB ** 2) ** 0.5
            return (A + B + C + D) / 4, max(LAB, LAD) * 1.4142135623731

        def precheck(r1, r2):
            c1, l1 = get_circle(r1)
            c2, l2 = get_circle(r2)
            d = numpy.sum((c1 - c2) ** 2) ** 0.5
            if d > l1 + l2:
                return False
            return True

        def is_overlap(r1, r2):
            if not precheck(r1, r2):
                return False
            p1 = Polygon(r1.tolist())
            p2 = Polygon(r2.tolist())
            ai = p1.intersection(p2).area
            a1 = p1.area
            a2 = p2.area
            if ai > 0.666666 * a1 and ai > 0.666666 * a2:
                return 100000000
            if ai > (0.6666666-uuu) * a1:
                return 2
            return 0

        fail = 0
        for i in range(len(rects))[::-1]:
            compare_cnt += 1
            fail += is_overlap(rect, rects[i])
            if fail > 1:
                break
        if not (fail > 1):
            rects.append(rect)
    return rects
very_cnt = 0
def remove_duplicated_rects_very_fast(rects, image_shape,uuu=0):
    global very_cnt
    print("remove_duplicated_rects_very_fast in")
    from time import clock
    tm = clock()
    image = numpy.zeros(image_shape, numpy.uint8)
    for rect in rects:
        cv2.fillPoly(image, numpy.array([rect]).astype('int32'), (255, 255, 255))
    # cv2.imwrite('..\\results\\' + '199601230000' + '\\' + '%d a.jpg' % very_cnt, image)
    group_cnt = 0
    for rect in rects:
        c = (rect[0] + rect[1] + rect[2] + rect[3]) // 4
        if c[1]>=image.shape[0] or c[0]>=image.shape[1]:
            continue
        if image[c[1], c[0], 0] == 255 and image[c[1], c[0], 1] == 255 and image[c[1], c[0], 2] == 255:
            group_cnt += 1
            r = (group_cnt >> (8 * 2)) & 255
            g = (group_cnt >> (8 * 1)) & 255
            b = (group_cnt >> (8 * 0)) & 255
            # r,g,b=random.randint(0,255),random.randint(0,255),random.randint(0,255)
            cv2.floodFill(image, None, (c[0], c[1]), (r, g, b))
        elif image[c[1], c[0], 0] == 0 and image[c[1], c[0], 1] == 0 and image[c[1], c[0], 2] == 0:
            print('that is not supposed to happen!')
    groups = [[] for i in range(group_cnt)]
    for rect in rects:
        c = (rect[0] + rect[1] + rect[2] + rect[3]) // 4
        if c[1]>=image.shape[0] or c[0]>=image.shape[1]:
            continue
        id = image[c[1], c[0], 0] * 256 * 256 + image[c[1], c[0], 1] * 256 + image[c[1], c[0], 2]
        id -= 1
        if id < 0 or id >= group_cnt:
            print('shit')
        groups[id].append(rect)
    if 0:
        for group in groups:
            r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            for rect in group:
                cv2.fillPoly(image, numpy.array([rect]).astype('int32'), (r, g, b))
        cv2.imwrite('..\\results\\' + '199601230000' + '\\' + '%d b.jpg' % very_cnt, image)
    very_cnt += 1
    ret = []
    for group in groups:
        ret += remove_duplicated_rects_fast(group,uuu)
    print('remove_duplicated_rects_very_fast out %f s\n' % (clock() - tm))
    # exit(0)
    return ret
def split_rect(r):
    A = r[0]
    B = r[1]
    C = r[2]
    D = r[3]
    AD = D - A
    AB = B - A
    LAD = numpy.sum(AD ** 2) ** 0.5
    LAB = numpy.sum(AB ** 2) ** 0.5
    if LAB / LAD > 1 / 3 and LAB / LAD < 3:
        return [r]
    ra = LAD / LAB
    if LAB > LAD:
        ra = LAB / LAD
        B, D = D, B
    ra = int(round(ra))
    ret = []
    for i in range(ra):
        v = (D - A) * ((i + 1) / ra)
        DD = A + v
        CC = B + v
        v = (D - A) * (i / ra)
        AA = A + v
        BB = B + v
        ret += [numpy.array([AA, BB, CC, DD]).astype('int64')]
    return ret
def split_rects(rects):
    ret = []
    for r in rects:
        ret += split_rect(r)
    return ret
def get_components(image, debug=0):
    print('get_components in')
    from time import clock
    tm = clock()
    salient = get_map_fast(image)
    salient_con = salient.copy()
    salient_con[salient_con < 255 * 0.5] = 0
    salient_con[salient_con >= 255 * 0.5] = 255
    salient[salient < 255 * 0.5] = 0
    salient[salient >= 255 * 0.5] = 255
    def is_good_rect(salient, rect):
        n = image.shape[0]
        cnt_all = 50
        cnt_in = 0
        A = rect[0]
        B = rect[1]
        C = rect[2]
        D = rect[3]
        AD = D - A
        AB = B - A
        LAD = numpy.sum(AD ** 2) ** 0.5
        LAB = numpy.sum(AB ** 2) ** 0.5
        if LAB < 4 or LAD < 4:
            return False
        # if LAD/LAB<1/3 or LAD/LAB>3:
        #    return False
        for i in range(cnt_all):
            u = random.uniform(0, 1)
            v = random.uniform(0, 1)
            P = A + u * AD + v * AB
            x = int(round(P[0]))
            y = int(round(P[1]))
            if x < 0:
                x = 0
            if x >= n:
                x = n - 1
            if y < 0:
                y = 0
            if y >= n:
                y = n - 1
            if salient[y, x] == 255:
                cnt_in += 1
        return cnt_in / cnt_all > 0.85
    if debug:
        rects_can, rects_image = mser_gray_refined(image, debug)
    else:
        rects_can = mser_gray_refined(image, debug)
    rects_can = [i for i in rects_can if is_good_rect(salient, i)]
    rects = remove_duplicated_rects_very_fast(rects_can, image.shape)
    if 0:
        rects_image_salient = image.copy()
        cv2.polylines(rects_image_salient, rects, 1, (255, 0, 0))
        cv2.imwrite('..\\results\\' + '199601230000' + '\\' + "salient %d" % (image.shape[0]) + '_0.jpg', salient)
        cv2.imwrite('..\\results\\' + '199601230000' + '\\' + "salient %d" % (image.shape[0]) + '_1.jpg',
                    rects_image_salient)
        # cv2.imwrite('..\\results\\' + '199601230000' + '\\' + "salient %d" % (image.shape[0]) + '_2.jpg', rects_image)
    print('get_components out %f s\n' % (clock() - tm))
    if debug:
        rects_image_salient = image.copy()
        cv2.polylines(rects_image_salient, rects, 1, (255, 0, 0))
        return salient, rects_image_salient, rects_image
    else:
        return rects, salient_con
def vector_len(v):
    return numpy.sum(v ** 2) ** 0.5
def vector_dot(u, v):
    return numpy.sum(u * v)
def vector_ang(u, v):
    if u[0] == 0 and u[1] == 0:
        return 0
    if v[0] == 0 and v[1] == 0:
        return 0
    t = vector_dot(u, v) / vector_len(u) / vector_len(v)
    if t > 1:
        t = 1
    if t < -1:
        t = -1
    return math.acos(t) / math.pi * 180
def vector_2_tuple(v):
    x = int(round(v[0]))
    y = int(round(v[1]))
    return (x, y)
def vector_cross(u, v):
    return u[0] * v[1] - u[1] * v[0]
class Rect:
    def __init__(self, r, i):
        self.v = r
        self.a = r[0]
        self.b = r[1]
        self.c = r[2]
        self.d = r[3]
        self.c = (r[0] + r[1] + r[2] + r[3]) / 4
        self.p = Polygon(r.tolist())
        self.s = self.p.area
        self.n = []
        self.i = i
        self.ad = self.d - self.a
        self.ab = self.b - self.a
        self.lad = numpy.sum(self.ad ** 2) ** 0.5
        self.lab = numpy.sum(self.ab ** 2) ** 0.5
        self.w = max(self.lab, self.lad)
        self.h = min(self.lab, self.lad)
    def max_project(self, d):
        return max([vector_dot(r - self.c, d) / vector_len(d) for r in self.v])
    def min_project(self, d):
        return min([vector_dot(r - self.c, d) / vector_len(d) for r in self.v])
    def max_project2(self, d):
        return max([vector_dot(r, d) / vector_len(d) for r in self.v])
    def min_project2(self, d):
        return min([vector_dot(r, d) / vector_len(d) for r in self.v])
    def project(self, d):
        t = [vector_dot(r - self.c, d) / vector_len(d) for r in self.v]
        return max(t) - min(t)
    def project2(self, d):
        t = [vector_dot(r, d) / vector_len(d) for r in self.v]
        return max(t) - min(t)
    def project3(self, d):
        t = [vector_dot(r, d) / vector_len(d) for r in self.v]
        return max(t), min(t)
    def get_angle(self, d):
        t = vector_dot(self.ab, d) / vector_len(d) / vector_len(self.ab)
        if t < 0:
            t = -t
        if t > 1:
            return 90
        t = math.acos(t) / math.pi * 180
        t = min(t, 90 - t)
        return t
    def get_face(self, d):
        for i in range(4):
            j = (i + 1) % 4
            k=(j+1)%4
            v1 = self.v[i] - self.c
            v2 = self.v[j] - self.c
            c1 = vector_cross(v1, d)
            c2 = vector_cross(v2, d)
            if (c1 <= 0 and c2 >= 0) or (c1 >= 0 and c2 <= 0):
                a = self.v[i]
                b = self.v[j]
                delta=vector_len(b-self.v[k])
                break
        v = a - b
        v[0], v[1] = -v[1], v[0]
        if vector_dot(d, v) < 0:
            v = -v
        return v,delta/2
def connect_check(r1, r2):
    if r1.c[0] == r2.c[0] and r1.c[1] == r2.c[1]:
        return False
    f1,d1 = r1.get_face(r2.c - r1.c)
    f2,d2 = r2.get_face(r2.c - r1.c)
    lf1 = vector_len(f1)
    lf2 = vector_len(f2)
    u1=d1*2 /lf1
    u2=d2*2/lf2
    if lf1 / lf2 > 1.6 or lf1 / lf2 < 1 / 1.6:
        return False
    if vector_ang(f1, f2) > 30:
        return False
    pf1=r1.c+f1/vector_len(f1)*d1
    pf2=r2.c-f2/vector_len(f2)*d2
    dis=vector_len(pf1-pf2)
    d1=r1.max_project(r2.c-r1.c)
    d2=r2.max_project(r1.c-r2.c)
    #if 1.0* (lf1+lf2)/2<dis:
    #    return False
    ff=(f1+f2)/2
    ddd1 = vector_dot(ff, pf1 - pf2) / vector_len(ff)
    ff[0],ff[1]=-ff[1],ff[0]
    ddd2=vector_dot(ff,pf1-pf2)/vector_len(ff)
    if abs(ddd1)>(1.75 if u1>2.3 or u2>2.3 else 0.6)* max(min(lf1,d1*2),min(lf2,d2*2)):
        return False
    if abs(ddd2)>(3/4 if u1>2.3 or u2>2.3 else 1/2)*max(lf1,lf2):
        return False
    if lf1 / r1.h > 2.3 and lf2 / r2.h > 2.3:
        return False
    if lf1 / r1.h > 3.3 or lf2 / r2.h > 3.3:
        return False
    '''if lf1 / r1.h > 2.5 or lf2 / r2.h > 2.5:
        if abs(ddd1) > 0.25 * (lf1 + lf2) / 2:
            return False
        if abs(ddd2) > 0.25 * (lf1 + lf2) / 2:
            return False'''
    def is_overlap(r1, r2):
        p1 = Polygon(r1.tolist())
        p2 = Polygon(r2.tolist())
        ai = p1.intersection(p2).area
        a1 = p1.area
        a2 = p2.area
        if ai > 0.5 * a1 or ai > 0.5 * a2:
            return True
    if is_overlap(r1.v,r2.v):
        return False
    return True
def get_chain(i, rects, chains, instack, u, stack, level=0):
    # print(i)
    instack[i] = 1
    stack.append(i)
    from copy import copy
    chains.append(copy(stack))
    for j in rects[i].n:
        if instack[j]:
            continue
        v = rects[j].c - rects[i].c
        if vector_ang(u, v) > 30:
            continue
        get_chain(j, rects, chains, instack, v, stack, level + 1)
        if level > 3:
            break
    stack.pop(-1)
    instack[i] = 0
def linefit(x, y):
    N = float(len(x))
    sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
    for i in range(0, int(N)):
        sx += x[i]
        sy += y[i]
        sxx += x[i] * x[i]
        syy += y[i] * y[i]
        sxy += x[i] * y[i]
    if sx * sx / N - sxx == 0:
        a = 1e1000
    else:
        a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
    b = (sy - a * sx) / N
    return a, b
def generate_words(rects):
    chains = []
    instack = [0] * len(rects)
    stack = []
    for i in range(len(rects)):
        get_chain(i, rects, chains, instack, numpy.array([0, 0]), stack)
    chains.sort(key=lambda x: len(x), reverse=True)
    used = set()
    new_rects = []
    for chain in chains:
        if len(used.intersection(set(chain))) != 0:
            continue
        used = used.union(set(chain))
        if len(chain) == 1:
            new_rects.append(rects[chain[0]])
            continue
        c = [rects[i].c for i in chain]
        cx = [i[0] for i in c]
        cy = [i[1] for i in c]
        a, b = linefit(cx, cy)
        if a != 1e1000:
            v = numpy.array([1, a])
        else:
            v = numpy.array([0.0, 1.0])
        u = numpy.array([-v[1], v[0]])
        h = 0
        for i in chain:
            h += rects[i].project2(u) / len(chain)
        minp = min([rects[i].min_project2(v) for i in chain])
        maxp = max([rects[i].max_project2(v) for i in chain])
        w = maxp - minp
        if a != 1e1000:
            x = minp * (1 / math.sqrt(1 + a * a))
            y = a * x
            x += -a * b / (a * a + 1)
            y += b / (a * a + 1)
        else:
            x = numpy.mean(cx)
            y = minp
        # print(v)
        v /= vector_len(v)
        u /= vector_len(u)
        tmp = numpy.array([x, y])
        A = tmp + u * h / 2
        B = tmp - u * h / 2
        C = B + v * w
        D = A + v * w
        new_rects.append(Rect(numpy.array([A, B, C, D]).astype('int64'), len(new_rects)))
    return new_rects
cvv = 0
def merge_rects(rects, contours, image):
    print('merge_rects in')
    from time import clock
    tm = clock()
    image_w, image_h = image.shape[:-1]
    rect_groups = [[(0, 0, 0)] for i in range(len(contours))]
    for i in range(len(contours)):
        rect_groups[i][0] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        rect_groups[i].append([])
    rects_ = []
    for rect in rects:
        A = rect[0]
        B = rect[1]
        C = rect[2]
        D = rect[3]
        AD = D - A
        AB = B - A
        dst = -1
        for i in range(100):
            u = random.uniform(0, 1)
            v = random.uniform(0, 1)
            P = A + u * AD + v * AB
            x = int(round(P[0]))
            y = int(round(P[1]))
            if x < 0:
                x = 0
            if x >= image_h:
                x = image_h - 1
            if y < 0:
                y = 0
            if y >= image_w:
                y = image_w - 1
            for j in range(len(contours)):
                t = contours[j]
                t = t.reshape((t.shape[0], t.shape[2]))
                bbPath = mplPath.Path(t)
                if bbPath.contains_point((x, y)):
                    dst = j
                    break
            if dst != -1:
                break
        if dst != -1:
            rects_.append(Rect(rect, len(rects_)))
            rect_groups[dst][1].append(rects_[-1])
    rects = rects_
    for rect_group in rect_groups:
        for r1 in rect_group[1]:
            for r2 in rect_group[1]:
                if connect_check(r1, r2):
                    r1.n.append(r2.i)
                    r2.n.append(r1.i)
    global cvv
    if 0:
        image = image.copy()
        for rect in rects:
            cv2.polylines(image, [rect.v], 1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),1)
        for r1 in rects:
            for i in r1.n:
                r2 = rects[i]
                cv2.line(image, vector_2_tuple(r1.c), vector_2_tuple(r2.c), (255, 0, 0), 1)
        cvv += 1
        cv2.imwrite('..\\results\\' + '199601230000' + '\\' + "%d" % (cvv) + '.jpg', image)
    rects = generate_words(rects)
    print('merge_rects out %f s\n' % (clock() - tm))
    return [i.v for i in rects]
def look_closer(image,salient,rects):
    ret=[]
    for rect in rects:
        mask = numpy.zeros(image.shape, numpy.uint8)
        cv2.fillPoly(mask, numpy.array([rect]).astype('int32'), (255, 255, 255))
        image233=cv2.bitwise_and(mask,image)
        ret.append(image233)
        image_w, image_h = image.shape[:-1]
        n = max(image_w, image_h)
        image_aug = numpy.zeros((n, n, 3)).astype('uint8')
        image_aug[:image_w, :image_h] = image233
        l = 0
        while (224 << l + 1) <= n:
            l += 1
        l += 1
        salients = []
        while l >= 0:
            image_aug = cv2.resize(image_aug, (224 << l, 224 << l))
            salient = cv2.resize(get_map_fast(image_aug),(n,n))
            salient[salient < 255 * 0.5] = 0
            salient[salient >= 255 * 0.5] = 255
            salients.append(salient)
            l -= 1
        salient = numpy.zeros((n, n))
        for s in salients:
            salient += s.astype('float32')
        salient[salient > 255] = 255
        salient = salient.astype('uint8')[:image_w, :image_h]
        ret.append(salient)
    return ret
def get_components_all_sizes(image, debug=0):
    print('get_components_all_sizes in')
    from time import clock
    tm = clock()
    image_w, image_h = image.shape[:-1]
    n = max(image_w, image_h)
    image_aug = numpy.zeros((n, n, 3)).astype('uint8')
    image_aug[:image_w, :image_h] = image
    l = 0
    while (224 << l + 1) <= n:
        l += 1
    l += 1
    salients = []
    rects = []
    while l >= 0:
        image_aug = cv2.resize(image_aug, (224 << l, 224 << l))
        result = get_components(image_aug)
        salients += [cv2.resize(result[1], (n, n))]
        # if debug:
        #    cv2.imwrite('..\\results\\' + '199601230000' + '\\' + "salient %d"%(224<<l) + '.jpg', cv2.resize(result[1], (n, n)))
        for rect in result[0]:
            rect = rect / (224 << l) * n
            rects.append(rect.astype('int64'))
        l -= 1
    rects = split_rects(rects)
    image_ = image.copy()
    for rect in rects:
        cv2.polylines(image_, [rect], 1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
    ret = []
    salient = numpy.zeros((n, n))
    for s in salients:
        salient += s.astype('float32')
    salient[salient > 255] = 255
    salient = salient.astype('uint8')[:image_w, :image_h]
    ret.append(salient)
    #ret.append(image_)
    rects = remove_duplicated_rects_very_fast(rects, image.shape)
    image_ = image.copy()
    for rect in rects:
        cv2.polylines(image_, [rect], 1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
    ret.append(image_)
    tmp, contours, hierarchy = cv2.findContours(salient.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = merge_rects(rects, contours, image)
    image_ = image.copy()
    for rect in rects:
        cv2.polylines(image_, [rect], 1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
    ret.append(image_)
    rects = merge_rects(rects, contours, image)
    image_ = image.copy()
    for rect in rects:
        cv2.polylines(image_, [rect], 1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
    ret.append(image_)
    rects = merge_rects(rects, contours, image)
    image_ = image.copy()
    for rect in rects:
        cv2.polylines(image_, [rect], 1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
    ret.append(image_)
    rects = merge_rects(rects, contours, image)
    image_ = image.copy()
    for rect in rects:
        cv2.polylines(image_, [rect], 1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
    ret.append(image_)
    rects = remove_duplicated_rects_very_fast(rects, image.shape,uuu=0.1666666666)
    image_ = image.copy()
    f=lambda r:   r.h>10 and (  r.s>2000 or r.w/r.h>4)
    rects=[r for r in rects if f(Rect(r,0))]
    for rect in rects:
        cv2.polylines(image_, [rect], 1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
    ret.append(image_)
    #ret+=look_closer(image,salient,rects)
    print('get_components_all_sizes out %f s\n' % (clock() - tm))
    if debug:
        return ret
    return rects,ret[-1]
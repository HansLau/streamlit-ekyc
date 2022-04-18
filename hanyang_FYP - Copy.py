import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import cv2
from recognize.crnn_recognizer2 import PytorchOcr
model_path=r'C:\Users\hans-\OneDrive - Asia Pacific University\Intern\bsimple_ocr\checkpoints\CRNN24_99_933.pth'
recognizer = PytorchOcr(model_path)
from math import *
from detect.ctpn_predict2 import get_det_boxes
import argparse
import re


def dis(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)

def sort_box(box):
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box

def charRec(img, text_recs, adjust=False):
    results = {}
    xDim, yDim = img.shape[1], img.shape[0]
    count = 0
    for index, rec in enumerate(text_recs):
        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])

        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # skewness calc

        partImg, coord = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)
        cv2.imwrite("output/image-" + str(count) + ".jpg", partImg)
        # dis(partImg)
        if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:  # filter out non-confirmity images
            continue
        text = recognizer.recognize(partImg)
        if len(text) > 0:
            results[index] = [rec]
            results[index].append(text)  
        
        count+=1

    return results

def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    
    paddingX = 0
    paddingY =0
    # cv2.imwrite("warp.jpg", imgRotation)
    # print( xdim,ydim)
    imgOut = imgRotation[max(1, int(pt1[1])) -paddingY : min(ydim - 1, int(pt3[1])) +paddingY,
             max(1, int(pt1[0])) -paddingX : min(xdim - 1, int(pt3[0])) + paddingX]
    
    coord = [max(1, int(pt1[0])) -paddingX , max(1, int(pt1[1])) -paddingY, min(xdim - 1, int(pt3[0])) + paddingX, min(ydim - 1, int(pt3[1])) +paddingY]
    
    #, min(xdim - 1, int(pt3[0])) + paddingX, max(1, int(pt1[1])) -paddingY , min(ydim - 1, int(pt3[1])) +paddingY]
    
    return imgOut, coord

def charRec2(img, text_recs,path, adjust=False):
    images_path = []
    coord_all = []
    results = {}
    xDim, yDim = img.shape[1], img.shape[0]
    count = 0
    for index, rec in enumerate(text_recs):
        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])

        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

        partImg, coord = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)
        Height, Width = partImg.shape[:2]
        partImg = cv2.resize(partImg, (int(Width/2), int(Height/2))) # * 3. Withotu division also ok
        cv2.imwrite(path + "/img_crop-" + str(count) + ".jpg", partImg)
        images_path.append(path + "/img_crop-" + str(count) + ".jpg")
        coord_all.append(coord)
        
        # dis(partImg)
        # if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:  # 过滤异常图片
        #    continue
        # text = recognizer.recognize(partImg)
        # if len(text) > 0:
        #    results[index] = [rec]
        #    results[index].append(text)  # 识别文字
        
        count+=1

    return images_path,coord_all
def ocr(image):
    #cv2.imwrite("test1.jpg", image)
    # detect
    text_recs, img_framed, image = get_det_boxes(image)
    cv2.imwrite("test2.jpg", image)
    text_recs = sort_box(text_recs)
    result = charRec(image, text_recs)
    return result, img_framed

def border(image):
    img = cv2.imread(image)
    (imageHeight, imageWidth) = img.shape[:2]

    border = cv2.copyMakeBorder(
        img,
        top = 100,
        bottom = 100,
        left = 75,
        right = 75,
        borderType = cv2.BORDER_CONSTANT,
        value = (255, 255, 255)
        )

    (borderHeight, borderWidth) = border.shape[:2]
    if borderHeight > 5000 or borderWidth > 5000:
        border = cv2.resize(border, (750, 1000), interpolation = cv2.INTER_AREA)

    return border
    
    
def match_outlines(orig_image, skewed_image, ii):
    # orig_image = np.array(orig_image)
    skewed_image = np.array(skewed_image)
    orig_image = np.array(orig_image)


 
    if ii == 1:
        #doesnt work anymore!
        # surf = cv2.xfeatures2d.SURF_create(400)

        surf = cv2.AKAZE_create(nOctaves=2,nOctaveLayers=2,descriptor_size=8) #nOctaves=4, nOctaveLayers=4, descriptor_size=61
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2   # or pass empty dictionary
    elif ii == 2:
        #surf = cv2.xfeatures2d.SIFT_create()
        surf = cv2.SIFT_create()
        bf = cv2.BFMatcher()
        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # search_params = dict(checks=50)   # or pass empty dictionary
    else :
        surf = cv2.ORB_create(5000, edgeThreshold=16, nlevels=5, scaleFactor=1.5 ,scoreType=cv2.ORB_FAST_SCORE) 
        #18000 #16000 #14000 #10000 #7000 
        # lower 7 complexity, works #nlevels=7, scaleFactor=1.5, edgeThreshold=12
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12 #6
                   key_size = 12,     # 20 #12
                   multi_probe_level = 1) #2
    kp1, des1 = surf.detectAndCompute(orig_image, None)
    kp2, des2 = surf.detectAndCompute(skewed_image, None)

    search_params = dict(checks=5) #50 #5
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # matches = bf.knnMatch(des1,des2, k=2) #matches = bf.match(des1,des2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # matches = sorted(matches, key = lambda n:n.distance)
    # p_matched.append(len(matches)/len(kp1))
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance: #0.7
            good.append(m)
    nn = 0
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance: #0.7
            matchesMask[i]=[1,0]
            nn = nn + 1
            
    #print "Good - {}".format(len(good)) 
    #if ii == 2:		
        #		
    #    print ("NN - {}".format(nn) )	
    
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
    img4 = cv2.drawMatchesKnn(orig_image,kp1,skewed_image,kp2,matches,None,**draw_params)		
    cv2.imwrite("output/matches.jpg", img4)
    # plt.imshow(img4,),plt.show()

    MIN_MATCH_COUNT = 8
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                              ]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                              ]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
        '''
        ss = M[0, 1]
        sc = M[0, 0]
        scaleRecovered = math.sqrt(ss * ss + sc * sc)
        thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
        
        print("MAP: Calculated scale difference: %.2f, "
                      "Calculated rotation difference: %.2f" %
                      (scaleRecovered, thetaRecovered))
        '''
        #deskew image
        im_out = cv2.warpPerspective(skewed_image, np.linalg.inv(M),
            (orig_image.shape[1], orig_image.shape[0]))
        return im_out, nn

    else:
        print("MAP: Not  enough  matches are found   -   %d/%d"
        			  % (len(good), MIN_MATCH_COUNT))
        return skewed_image , nn

def ocr2(image):
    text = recognizer.recognize(image)
    return text
    

def single_pic_proc(image_file):
    image = np.array(Image.open(image_file).convert('RGB'))
    result, image_framed = ocr(image)
    return result,image_framed
    
def single_pic_proc2(image_file):
    image = np.array(Image.open(image_file).convert('RGB'))
    result= ocr2(image)
    return result

def dis(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

'''
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
    help="first input image")	
args = vars(ap.parse_args())
'''
#imgPath = args["first"]
#imgPath = imgPath.replace('"','')

def resizeSave(imgPath):
    lowQuality = cv2.imread(imgPath)
    Height, Width = lowQuality.shape[:2]
    resized=0

    while ( (Width<895) or (Height<565) ):
        lowQuality = cv2.resize(lowQuality, (int(Height * 2) ,int(Width * 2) ))
        Height, Width = lowQuality.shape[:2]
        resized=1

    if resized == 1:
        name = imgPath.split(".")
        imgPath = name[0]+"resize.jpg"
        cv2.imwrite(imgPath, lowQuality)	

    return imgPath   

def slice(img, number_tiles):
    im_h, im_w = img.shape[:2]
    columns = 0
    rows = 0
    if number_tiles:
        columns = int(ceil(sqrt(number_tiles)))
        rows = int(ceil(number_tiles / float(columns)))

    tile_w, tile_h = int(floor(im_w / columns)), int(floor(im_h / rows))

    tiles = []
    #number = 1
    for pos_y in range(0, im_h - rows, tile_h):  # -rows for rounding error.
        for pos_x in range(0, im_w - columns, tile_w):  # as above.
            image = img[pos_y:pos_y+tile_h , pos_x:pos_x + tile_w].copy()
            #position = (int(floor(pos_x / tile_w)) + 1, int(floor(pos_y / tile_h)) + 1)
            #coords = (pos_x, pos_y)
            #tile = Tile(image, number, position, coords)
            tiles.append(image)
            #number += 1
    return tuple(tiles)


def quantifier(img):
    '''
    image resolution affects result. <100 px immediately = bad
    '''
    num_tiles = 32
    cv2.imwrite("crppped.jpg", img)

    tiles = slice(img, num_tiles)
    val = []
    for tile in tiles:
        #img = cv2.imread(tile.filename, 0)
        blurred = cv2.GaussianBlur(tile, (3, 3), 0)
        # compute the median
        # mi = np.median(tile)   
        # sigma = 0.33
        # # computer lower & upper thresholds 
        # lower = int(max(0, (1.0 - sigma) * mi))
        # upper = int(min(255, (1.0 + sigma) * mi))
        # edges = cv2.Canny(blurred, lower, upper)
        
        #edges = cv2.Canny(blurred, 100, 200) #Perhaps too relaxed
        edges = cv2.Canny(blurred, 240, 250) #Stricter

        avg  = np.average(edges)
        
        val.append(avg)
    
    res = ''
    blr = 0
    # 40%
    pct = round(len(val) * 0.40)
    
    print('Threshold: ' + str(pct))
    
    for i in range(0, len(val)):
        #print(vall[i])
        if val[i] <= 0:
            blr += 1
    print('Blur' + str(blr))
    
    if blr > pct:
        res = 630 #560
    else:
        res = 390 #390
    #print(res, blr)    
    return res

class rHelperFunc(object):
    '''
    Given a pixel coordinate (for original image with h=560), calculates new pixel coordinate for after image has been resized.
    Uses ratio and stuff
    '''
    newHeight = 0
    originalHeight = 0

    def __call__(self, pixel):
        result = int ( (pixel/rHelperFunc.originalHeight)* rHelperFunc.newHeight )
        #print("New res: {} New coord: {} original: {}" .format(rHelperFunc.newHeight, result, pixel))
        return result

def start(imgPath):
    name = imgPath.split(".")
    directoryName = os.path.dirname(name[0])    

    # image = np.array(Image.open(imgPath)) #.convert('RGB'))
    image = np.array(cv2.imread(imgPath))
    #result, image_framed = ocr(image)
    start_time = time.time()

    # resolution = 630 #quantifier(image[15:80,100:280].copy())
    # print("resolution = " + str(resolution))
    rHelperFunc.newHeight = 550
    rHelperFunc.originalHeight = 550
    rHelper = rHelperFunc()
    
    IC = image[ 97:172, 20:375]
    Gender = image[ 445:500, 540:830]
    Details = image[ 320:520, 25:455]
    face = image[ 124:400, 546:786]
    cv2.imwrite("output/ic_face.jpg", face)
    print(IC.shape[:],Details.shape[:],Gender.shape[:])

    IC = np.pad(IC,[ (0, 0), (0, 75),(0, 0)] , mode='constant')
    Gender = np.pad(Gender,[ (0, 0), (0, 140),(0, 0)] , mode='constant')
    image = np.concatenate((IC,Gender,Details), axis=0)

    text_recs, img_framed, image = get_det_boxes(image,550,expand=True ) #550
    cv2.imwrite(directoryName + "/test2.jpg", img_framed)	

    #print("--- %s seconds for detection---" % (time.time() - start_time))

    text_recs = sort_box(text_recs)
    res,coord_all = charRec2(image, text_recs, directoryName)

    name, id_num, add, religion, gender = "","","","", ""
    nt1, nt2 = "",""
    it1, it2 = "",""
    at1, at2 =""," "
    add1, add2 = "",""
    print("Text Boxing time: {}" .format(time.time() -  start_time))

    nameFound = 0
    nameBottomY = 0
    #Dont OCR First 2 results [2:]
    for i in range(len(res)):
        #print(res[i])
        # imgP = cv2.imread(res[i])
        coord = coord_all[i]        
        text2 = single_pic_proc2(res[i])
        print(coord,text2)
        
        # Number
        if coord[1] <= rHelper(105):    
            id_num = text2
        #Detail
        elif coord[3]<225:
            if coord[1]>=147 and coord[2] >= 340:
                gender = text2
            elif coord[1]>=147:
                religion = text2
            else:
                it1 = text2
        #Name + Address
        else:
            #if bottom y of 1st box close to top y of 2nd box, is NAME
            if nameFound==0:
                if nameBottomY == 0:
                    name += text2
                    nameBottomY = coord[3]
                elif abs(nameBottomY-coord[1])<=25:
                    name += " " + text2
                    nameFound = 1
                else:
                    add= text2
                    nameFound =1
            else:
                if add == "":
                    add = text2
                else:
                    add += ", " + text2
        it1 = it1 + "^" + text2
        # name id_num add
    print ("id:{}".format( id_num))
    print ("religion:{}".format( religion))
    print ("gender:{}".format( gender))
    print ("name:{}".format( name))
    print ("add:{}".format( add))
    

total_time = time.time()    
dir_path = os.path.dirname(os.path.realpath(__file__))
#Input Image
#imgPath = "output/2aa6a732-f4c5-4014-bfd8-a287c656b151.jpg"
#imgPath = "pr_back.jpg"
imgPath = "input/test2.jpg"
name = imgPath.split(".")
directoryName = os.path.dirname(name[0])
    
#imgP = cv2.imread(imgPath) 
#Low quality inputs, resize and save before doing code  as is
newImgPath = resizeSave(imgPath)
imgP = border(newImgPath)
(waldoHeight, waldoWidth) = imgP.shape[:2]
 
flag1=0
img1 = cv2.imread(dir_path + "/SIM2.jpg")
image1, dat1 = match_outlines(img1, imgP, 3)#1

flag1 = dat1
image1 = cv2.resize(image1, (840, 530))
newPath = dir_path +"/output/ID_corrected.jpg"
cv2.imwrite(newPath, image1)	

template_time = total_time - time.time()
print("Template time: {}" .format(template_time))

start(newPath)	
print("Template time: {}" .format(template_time))
print("Total time: {}" .format(time.time() - total_time ))

print("Original img size " + str(imgP.shape[:2]) + ", resized to 840,530")
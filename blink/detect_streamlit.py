
import cv2
import mediapipe as mp
import math
import numpy as np
# variables 
# constants
MOUTH_OPEN_FRAME =3
FONTS =cv2.FONT_HERSHEY_COMPLEX
PINK = (147,20,255)
GREEN = (0,255,0)

LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 

# landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv2.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

def findEuclideanDistance(source_representation, test_representation): #distance.findEuclideanDistance
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def openRatio(img, landmarks, lips_indices):
    lips_top = landmarks[lips_indices[0]]
    lips_bottom = landmarks[lips_indices[10]]
    lips_left = landmarks[lips_indices[34]]
    lips_right = landmarks[lips_indices[16]]
    #61 left lip, 291 right lip, 13 top light, 14 bottom lip
    #0, 10, 34, 16

    lipsvDistance = findEuclideanDistance(np.array(lips_top), np.array(lips_bottom))
    lipshDistance = findEuclideanDistance(np.array(lips_right), np.array(lips_left))
    lipsRatio = lipshDistance/lipsvDistance

    return lipsRatio 


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

# def detectMouth(name):
    
#     CEF_COUNTER =0
#     TOTAL_OPENS =0
#     timer = 0
#     map_face_mesh = mp.solutions.face_mesh
#     camera = cv2.VideoCapture(0)

#     with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
#         # starting Video loop here.
#         while True:
#             timer+=1
#             ret, frame = camera.read() # getting frame from camera 
#             if not ret: 
#                 break # no more frames break
            
#             #  resizing frame
#             frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
#             frame_height, frame_width= frame.shape[:2]
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#             results  = face_mesh.process(rgb_frame)
#             if results.multi_face_landmarks:
#                 mesh_coords = landmarksDetection(frame, results, False)
#                 ratio = openRatio(frame, mesh_coords, LIPS)
#                 cv2.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, GREEN, 2)
#                 # utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)

#                 if ratio >0.26: #0.15 for smile
#                     CEF_COUNTER +=1
#                     cv2.putText(frame, 'Mouth Open', (200, 50), FONTS, 1.3, PINK, 2)
#                     # utils.colorBackgroundText(frame,  f'Mouth Open', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )

#                 else:
#                     if CEF_COUNTER>MOUTH_OPEN_FRAME:
#                         TOTAL_OPENS +=1
#                         CEF_COUNTER =0
#                         output = "output/mouth_" + name + ".jpg"
#                         cv2.imwrite(output, frame)
#                         camera.release()
#                         cv2.destroyAllWindows()
#                         return True
#                         break
#                 cv2.putText(frame, f'Total Opens: {TOTAL_OPENS}', (100, 150), FONTS, 0.6, GREEN, 2)
#                 # utils.colorBackgroundText(frame,  f'Total Opens: {TOTAL_OPENS}', FONTS, 0.7, (30,150),2)
#                 cv2.polylines(frame,  [np.array([mesh_coords[p] for p in LIPS ], dtype=np.int32)], True, GREEN, 1, cv2.LINE_AA)

#             cv2.imshow('frame', frame)
#             key = cv2.waitKey(2)
#             if key==ord('q') or key ==ord('Q'):
#                 break
#             if timer >600:
#                 break
#         cv2.destroyAllWindows()
#         camera.release()
#         return False


from  ic_module import *
from blink.detect import detectMouth
from deepface import DeepFace

dir_path = os.path.dirname(os.path.realpath(__file__))

def ic_detect(imgPath):
    #####################
    ####IC Detection and OCR
    # imgPath = "input/test2.jpg"

    #IC Template Matching
    total_time = time.time()    
    imgP = border(imgPath)
    img1 = cv2.imread(dir_path + "/SIM2.jpg")
    image1, flag1 = match_outlines(img1, imgP, 3)#1
    image1 = cv2.resize(image1, (840, 530))
    newPath = dir_path +"/output/ID_corrected.jpg"
    cv2.imwrite(newPath, image1)	

    #OCR, Face Extraction
    template_time = total_time - time.time()
    print("----------Template Matching time: {}" .format(template_time))
    print()
    ocr = start(newPath)	
    print("---------IC Text Extraction and OCR time: {}" .format(time.time() - total_time - template_time ))
    print()
    # print("Original img size " + str(imgP.shape[:2]) + ", resized to 840,530")

    name = ocr["name"]
    return name, ocr

def face_verification(icImg, camImg):
    #######################
    ##Verification
    # icImg = dir_path+'/output/ic_face.jpg'
    # camImg = dir_path+'/real_face/dad (2).jpg'

    #Compare IC image against camera img for Verification
    total_time = time.time()
    

    detector_backend = 'retinaface' # 'mediapipe' WITHOUT ALIGN=TRUE, 'opencv'
    result = DeepFace.verify(img1_path = camImg, img2_path = icImg, model_name = "ArcFace", distance_metric = "euclidean", enforce_detection = False, detector_backend = detector_backend ,  normalization = 'ArcFace')
    print("Verification distance", result['verified'], result['distance'])
    
    verification_time = total_time - time.time()
    print("-------Verification time: {}" .format(verification_time))

    return  result['verified'], result['distance']


def ekyc(icPic, camPic, webcamMode = False):
    #####
    #WEBCAM = True overrides previous pictures, allows user to upload/take their own pic
    #WEBCAM is made for Streamlit!
    #  
    name, ocr = ic_detect(icPic)
    ic_face_pic = "output/" + ocr['savedface'] + ".jpg"
    verification, distance= face_verification(icImg=ic_face_pic, camImg= camPic)
    detect = detectMouth(name)

    #############
    #Evaluation
    import time
    time.sleep(1) 
    print()
    print("::::::::::::::e-KYC RESULTS:::::::::::::::::::::")
    if not webcamMode: print(icPic)

    print("VERIFICATION:")
    if verification==True:
        print("PASSED, Distance: ", distance)
    else:
        print("FAILED, Distance: ", distance)

    print("\nOPEN MOUTH DETECTION:")
    if detect==True:
        print("PASSED")
    else:
        print("FAILED or Timed out ")
    
    print(ocr) #pprint



ekyc(icPic = dir_path+"/real_face/dad3.jpg", camPic = dir_path+'/real_face/dad (7).jpg', webcamMode = False)
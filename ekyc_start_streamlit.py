import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

from  ic_module import *
from blink.detect_streamlit import  * #detectMouth,
from deepface import DeepFace


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

MOUTH_OPEN_FRAME =3
FONTS =cv2.FONT_HERSHEY_COMPLEX
PINK = (147,20,255)
GREEN = (0,255,0)

LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
dir_path = os.path.dirname(os.path.realpath(__file__))
# dir_path = os.path.dirname(os.path.abspath(__file__))

DEMO_VIDEO = 'demo.mp4'
DEMO_IC = "input/hy.jpg"
DEMO_SELFIE = "real_face/hy (7).jpg"
DEMO_IC_FACE = "face/hy.jpg"
DEMO_IMAGE = 'demo.jpg'
name = "unknown"
ocr = dict()

st.sidebar.title('e-KYC System')


st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def ic_detect(img):
    #####################
    ####IC Detection and OCR
    # imgPath = "input/test2.jpg"

    #IC Template Matching
    total_time = time.time()    
    imgP = border(img)
    img1 = cv2.imread(dir_path + "/SIM2.jpg")
    image1, flag1 = match_outlines(img1, imgP, 3)#1
    image1 = cv2.resize(image1, (840, 530))
    newPath = dir_path +"/output/ID_corrected.jpg"
    cv2.imwrite(newPath, image1)	

    #OCR, Face Extraction
    results = []
    template_time =  time.time() - total_time
    toPrint = "-Template Matching time: " + str(template_time)
    results.append(toPrint)
    st.text(toPrint)
    
    ocr = start(newPath)	

    toPrint = "-IC Text Extraction and OCR time: " + str(time.time() - total_time - template_time )
    results.append(toPrint)
    st.text(toPrint)
    
    st.session_state['ic_detect_time'] = results
    # print("Original img size " + str(imgP.shape[:2]) + ", resized to 840,530")

    name = ocr["name"]
    idCorrected = newPath.split(".")[0] + "_" +  name + ".jpg"

    try:
        os.rename(newPath, idCorrected)
    except WindowsError:
        os.remove(idCorrected)
        os.rename(newPath, idCorrected)

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
    
    results = []
    verification_time = total_time - time.time()
    toPrint = "-Verification time: " + str(verification_time)
    results.append(toPrint)
    toPrint = "Verification Pass: " + str(result['verified'])
    results.append(toPrint)
    toPrint = "Verification distance: " + str(result['distance'])
    results.append(toPrint)

    st.session_state['face_verification_results'] = results
    
    # st.text(toPrint)   

    return  result['verified'], result['distance']

# @st.cache(allow_output_mutation=True, suppress_st_warning =True)
def detectMouth(name, detection_confidence, tracking_confidence, ratioThreshold = 0.26, max_faces=1):
    fps = 0
    i = 0
    frame=np.zeros(shape=[512, 512, 3], dtype=np.uint8)
    with mp_face_mesh.FaceMesh(
    min_detection_confidence=detection_confidence,
    min_tracking_confidence=tracking_confidence , 
    max_num_faces = max_faces) as face_mesh:
        prevTime = 0
        CEF_COUNTER =0
        TOTAL_OPENS =0
        detected = False

        while vid.isOpened():
            i +=1
            ret, frame = vid.read()
            # if not ret:
            #     continue
            if ret == False:
                break
            
            frame = image_resize(image = frame, width = 640)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            face_count = 0
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    face_count += 1

                    mesh_coords = landmarksDetection(frame, results, False)
                    ratio = openRatio(frame, mesh_coords, LIPS)
                    cv2.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, GREEN, 2)

                    if ratio > ratioThreshold: #0.15 for smile
                        CEF_COUNTER +=1
                        cv2.putText(frame, 'Mouth Open', (200, 50), FONTS, 1.3, PINK, 2)
                        # utils.colorBackgroundText(frame,  f'Mouth Open', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )

                    else:
                        if CEF_COUNTER>MOUTH_OPEN_FRAME:
                            TOTAL_OPENS +=1
                            CEF_COUNTER =0
                            output = "output/mouth_" + name + ".jpg"
                            cv2.imwrite(output, frame)
                            detected = True
                            vid.release()
                            return detected, frame

                    cv2.putText(frame, f'Total Opens: {TOTAL_OPENS}', (100, 150), FONTS, 0.6, GREEN, 2)
                    # utils.colorBackgroundText(frame,  f'Total Opens: {TOTAL_OPENS}', FONTS, 0.7, (30,150),2)
                    cv2.polylines(frame,  [np.array([mesh_coords[p] for p in LIPS ], dtype=np.int32)], True, GREEN, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'No Face', (200, 50), FONTS, 1.3, PINK, 2)
            
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            # if record:
            #     #st.checkbox("Recording", value=True)
            #     out.write(frame)
            #Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
            # frame = image_resize(image = frame, width = 640, inter=cv2.INTER_CUBIC)
            stframe.image(frame,channels = 'BGR',use_column_width=True)
            if i >600:
                break
        # vid.release()
        # return detected, frame
    vid.release()
    return detected, frame

app_mode = st.sidebar.selectbox('Choose Menu',
['e-KYC Introduction and Explanation','Perform e-KYC Verification']
)

st.sidebar.subheader('Input Images')

if app_mode =='e-KYC Introduction and Explanation':
    st.title('e-KYC System with Anti-Spoofing')
    st.markdown('**StreamLit** Web GUI for integrated e-KYC verification model with anti-spoofing!')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    st.markdown('''
          # My name is  Lau Han Yang  (TP051144). \n
           
           e-KYC pipeline consists of obtaining images of Detection, Extraction and Verification. \n
           1) IC Card is first detected and then Text and IC Face is extracted using ORB Feature Matching and a trained CTPN+CRNN Text Detection model.\n
           2) User's Selfie Image is then verified with actual User Selfie picture using ArcFace Facial Verification for anti-spoofing. \n
           3) Mouth Opening is used for anti-spoofing, which is suitable for low-end devices with low-quality camera\n
         \n
            ''')

elif app_mode =='Perform e-KYC Verification':

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

    ccol1, ccol2 = st.columns(2)
    # st.title()
    with ccol1:
        icimg_file_buffer = st.file_uploader("1) Upload IC Image for OCR and IC Face Extraction", type=[ "jpg", "jpeg",'png'])
        st.markdown("ORB Feature Matching and \nCTPN+CRNN Text Detection")
    with ccol2:
        selfieimg_file_buffer = st.file_uploader("2) Upload Selfie Image for verification against IC Face", type=[ "jpg", "jpeg",'png'])
        st.markdown("ArcFace Face Verification against \nIC Face")

    if icimg_file_buffer is not None:
        icimg = np.array(Image.open(icimg_file_buffer))
    else:
        demo_image = DEMO_IC
        icimg = np.array(Image.open(demo_image))

    if selfieimg_file_buffer is not None:
        selfieimg = np.array(Image.open(selfieimg_file_buffer))
    else:
        demo_image = DEMO_SELFIE
        selfieimg = np.array(Image.open(demo_image))

    st.session_state['icimg'] = icimg
    st.session_state['selfieimg'] = selfieimg
    if 'ic_face_pic' not in st.session_state:
        st.session_state['ic_face_pic'] = DEMO_IC_FACE   
    if 'ic_button' not in st.session_state:
            st.session_state['ic_button'] = False
    if 'ocr_success' not in st.session_state:
        st.session_state['ocr_success'] = False

    st.sidebar.text('IC Image')
    st.sidebar.image(icimg)
    st.sidebar.text('Selfie Image')
    st.sidebar.image(selfieimg)

    col1, col2 = st.columns(2)
    ####
    ##EXTRACT IC with ORB, PERFORM OCR and EXTRACT IC FACE
    with col1:
        ic_button = st.button('Extract IC')
        if ic_button:
            st.session_state['ic_button'] = True
            with st.spinner("IC Detection, Perspective Correction, OCR in process"):
                name, ocr = ic_detect(cv2.cvtColor(icimg, cv2.COLOR_RGB2BGR))

                if name != "" and name != "unknown":
                    st.session_state['ocr_success'] = True
                    st.session_state['ocr'] = ocr 
                    st.session_state['name'] = name 
                    ic_face_pic = "output/" + ocr['savedface'] + ".jpg"
                    st.session_state['ic_face_pic'] = ic_face_pic
                else:
                    st.session_state['ocr_success'] = False
                
        ###
        #STREAMLIT LIMITATION. OUTPUTTING RESULTS FROM PREVIOUS RUN
        st.sidebar.text('IC Face Image')
        st.sidebar.image(st.session_state['ic_face_pic'])
        if 'ic_detect_time' not in st.session_state:
            st.session_state['ic_detect_time'] = []

        if st.session_state['ocr_success']:
            for i in st.session_state['ic_detect_time']:
                st.text(i)
            st.text(" ")

            if 'ocr' not in st.session_state:
                st.session_state['ocr'] = ocr 
            for k, v in st.session_state['ocr'].items():
                out = str(k) + ": " + str(v)
                st.text(out)
        else:
            if st.session_state['ic_button']:
                st.text("Failed. Card not extracted from image, no IC details found")

    ######
    ###Face Verification using ArcFace against IC Face
    with col2:
        st.checkbox("IC Face Extracted", value=(st.session_state['ocr_success'] and st.session_state['ic_button']), disabled=True)
        st.write("Will use DEMO_IC_FACE if not True")
        verify_button = st.button('Verify IC with Selfie Face')
        if verify_button:
            with st.spinner("Verification with ArcFace module"):
                verification, distance= face_verification(icImg=st.session_state['ic_face_pic'], camImg= selfieimg) #selfie img can change too but works oh well
        
        #STREAMLIT LIMITATION. OUTPUTTING RESULTS FROM PREVIOUS RUN
        if 'face_verification_results' not in st.session_state:
            st.session_state['face_verification_results'] = ""
        for i in st.session_state['face_verification_results']:
            st.text(i)



####################
    #########
    # Mouth Open Detection Anti-SPoofing
    st.markdown("---")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    use_webcam = st.checkbox('Use Webcam for Mouth Open Detection Anti-Spoofing')

    video_file_buffer = False
    if not use_webcam:
        video_file_buffer = st.file_uploader("3) Upload a video for Mouth Open Detection Anti-Spoofing", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    
    mouthStart_button = st.button("START MOUTH ANTI-SPOOFING")
    # st.markdown(' ## Output')
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    # detectFaces doesnt support currently
    # max_faces = st.number_input('Maximum Number of Faces', value=1, min_value=1)
    column1, column2 = st.columns(2)
    with column1:
        detection_confidence = st.slider('Face Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    with column2:
        ratio = st.slider('Mouth Open Ratio Threshold', min_value =0.0,max_value = 0.5,value = 0.26)
    # tracking_confidence = st.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = 0.5 #pointless
    
    stframe = st.empty()
    
    frametest=np.zeros(shape=[512, 512, 3], dtype=np.uint8)
    tfflie = tempfile.NamedTemporaryFile(delete=False)
    if mouthStart_button:
        if not video_file_buffer:
            if use_webcam:
                vid = cv2.VideoCapture(0)
            else:
                vid = cv2.VideoCapture(DEMO_VIDEO)
                tfflie.name = DEMO_VIDEO
        
        else:
            tfflie.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tfflie.name)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))
        
        st.sidebar.text('Input Video')
        st.sidebar.video(tfflie.name)

        detected, frametest = detectMouth(name, detection_confidence, tracking_confidence, ratio)
        st.session_state['mouth_detected'] = detected
        st.session_state['mouth_open_frame'] = frametest ##? If none detected?
        stframe.empty()

    if 'mouth_detected' not in st.session_state:
        st.session_state['mouth_open_frame'] = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
        st.session_state['mouth_detected'] = False

    if st.session_state['mouth_detected']:
        st.text('Mouth Opened, anti-spoofing passed:')
        # st.subheader('Output Image')
        st.image(st.session_state['mouth_open_frame'],channels = 'BGR',use_column_width= True)
    else:
        st.text("TIMED OUT: NO MOUTH FOUND")
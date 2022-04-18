#Modified by Augmented Startups 2021
#Face Landmark User Interface with StreamLit
#Watch Computer Vision Tutorials at www.augmentedstartups.info/YouTube
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

MOUTH_OPEN_FRAME =3
FONTS =cv2.FONT_HERSHEY_COMPLEX
PINK = (147,20,255)
GREEN = (0,255,0)

LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 


DEMO_VIDEO = 'demo.mp4'
DEMO_IMAGE = 'demo.jpg'
name = "unknown"

st.title('Face Mesh Application using MediaPipe')

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

st.sidebar.title('Face Mesh Application using MediaPipe')
st.sidebar.subheader('Parameters')

@st.cache()
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


# landmark detection function 
@st.cache(allow_output_mutation=True)
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

# @st.cache(allow_output_mutation=True, suppress_st_warning =True)
def detectMouth(name, detection_confidence, tracking_confidence, max_faces=1):
    fps = 0
    i = 0
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

                    if ratio >0.26: #0.15 for smile
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
            ### Streamlit very slow.. reduce to 400from 600
            if i >400:
                break
        # vid.release()
        # return detected, frame
    vid.release()
    return detected, frame

app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Run on Image','Run on Video']
)

if app_mode =='About App':
    st.markdown('In this application we are using **MediaPipe** for creating a Face Mesh. **StreamLit** is to create the Web Graphical User Interface (GUI) ')
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
    st.video('https://www.youtube.com/watch?v=FMaNNXgB_5c&ab_channel=AugmentedStartups')

    st.markdown('''
          # About Me \n 
            Hey this is ** Ritesh Kanjee ** from **Augmented Startups**. \n
           
            If you are interested in building more Computer Vision apps like this one then visit the **Vision Store** at
            www.augmentedstartups.info/visionstore \n
            
            Also check us out on Social Media
            - [YouTube](https://augmentedstartups.info/YouTube)
            - [LinkedIn](https://augmentedstartups.info/LinkedIn)
            - [Facebook](https://augmentedstartups.info/Facebook)
            - [Discord](https://augmentedstartups.info/Discord)
        
            If you are feeling generous you can buy me a **cup of  coffee ** from [HERE](https://augmentedstartups.info/ByMeACoffee)
             
            ''')
elif app_mode =='Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    # use_webcam = st.button('Use Webcam')
    use_webcam = st.checkbox('Use Webcam')
    # record = st.sidebar.checkbox("Record Video")
    # if record:
    #     st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
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

    # st.slider('test',  min_value = 0.0,max_value = 1.0,value = 0.5)
    video_file_buffer = False
    if not use_webcam:
        video_file_buffer = st.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    
    mouthStart = st.button("START MOUTH ANTI-SPOOFING")
    
    st.markdown(' ## Output')
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

    st.markdown("<hr/>", unsafe_allow_html=True)
    
    
    # detectFaces doesnt support currently
    # max_faces = st.number_input('Maximum Number of Faces', value=1, min_value=1)
    detection_confidence = st.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    tracking_confidence = st.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)
    
    stframe = st.empty()


    tfflie = tempfile.NamedTemporaryFile(delete=False)
    if mouthStart:
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

        #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        # codec = cv2.VideoWriter_fourcc('V','P','0','9')
        # out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))
        
        st.sidebar.text('Input Video')
        st.sidebar.video(tfflie.name)

        detected, frame = detectMouth(name, detection_confidence, tracking_confidence)
        # out.release()

        stframe.empty()
        if detected:
            st.text('Mouth Open Instance')
            # st.subheader('Output Image')
            st.image(frame,use_column_width= True)
        else:
            st.text("NOTHING FOUND")


    

elif app_mode =='Run on Image':

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    st.sidebar.markdown('---')

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
    st.markdown("**Detected Faces**")
    kpi1_text = st.markdown("0")
    st.markdown('---')

    max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    st.sidebar.markdown('---')

    icimg_file_buffer = st.sidebar.file_uploader("Upload IC Image", type=[ "jpg", "jpeg",'png'])
    selfieimg_file_buffer = st.sidebar.file_uploader("Upload Selfie Image", type=[ "jpg", "jpeg",'png'])

    if icimg_file_buffer is not None:
        icimg = np.array(Image.open(icimg_file_buffer))
    else:
        demo_image = DEMO_IMAGE
        icimg = np.array(Image.open(demo_image))

    if selfieimg_file_buffer is not None:
        selfieimg = np.array(Image.open(selfieimg_file_buffer))
    else:
        demo_image = DEMO_IMAGE
        selfieimg = np.array(Image.open(demo_image))

    st.sidebar.text('IC Image')
    st.sidebar.image(icimg)
    st.sidebar.text('Selfie Image')
    st.sidebar.image(selfieimg)
    face_count = 0

    # Dashboard
    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=max_faces,
    min_detection_confidence=detection_confidence) as face_mesh:

        results = face_mesh.process(image)
        out_image = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            face_count += 1

            #print('face_landmarks:', face_landmarks)

            mp_drawing.draw_landmarks(
            image=out_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
        st.subheader('Output Image')
        st.image(out_image,use_column_width= True)
# Watch Tutorial at www.augmentedstartups.info/YouTube
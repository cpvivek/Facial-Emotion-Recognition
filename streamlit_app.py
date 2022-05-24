import cv2
import numpy as np
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.utils import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode

# load model
emotion_dict = {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy', 4: 'sad',5: 'surprised', 6: 'neutral'}
# load json and create model
json_file = open('model_fer_4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("model_fer_4.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    
    pages=["Home","About"]
    
    with st.sidebar:
        st.title("select")
        page_name=st.selectbox('select page:', pages)
                 
    st.title(page_name)
                 
    if page_name=='Home':
    # Face Emotion Application #
        home_html = """
        <body style="background-color:blue;">
        <div style="background-color:red ;padding:10px">
        <h2 style="color:white;text-align:center;">Face Emotion Recognisation App</h2>
        <style>#"An Application by Vivek CP" {text-align: center}</style>
        </div>
        </body>
        """


        st.markdown(home_html, unsafe_allow_html=True)
        st.write("Project by Vivek CP")
        st.write("Facial Emotion Recognition in Real Time")
        st.write("**Instructions**")
        st.write('''

                    Click on START, and grant access to webcam to start the program. The program will try to predict your emotion in real time.
                    Click on STOP to end the session

                    ''')
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                            video_processor_factory=Faceemotion)
       
    elif page_name=='About':
        about_html="""<body>
        <h4 style="font-size:30px"> Facial emotion detection using CNN in real time</h4>
        <body>""" 
        
        
        st.markdown(about_html,unsafe_allow_html=True)
        st.write("This project is developed by Vivek CP, Data Science Trainee at AlmaBetter.")
        statement_html="""<body>
        <h4 style="font-size:20px"> About Project </h4>
        <p> Facial emotion recognition is an age old problem in the field of deep learning. The learning objective of the project is to gain hands on experience of developing a CNN model, and deploy it in real time. The model has achieved an training accuracy of 74% and test accuracy of 67%. The numbers can be improved by expanding the dataset size and using techniques like transfer learning. This would be included in the future scope of the project. <p>
        <body>"""
        
        st.markdown(statement_html,True)
            
            
            
    else:
        pass


if __name__ == "__main__":
    main()

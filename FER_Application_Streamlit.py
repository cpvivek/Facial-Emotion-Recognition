#importing libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import img_to_array
import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration


# In[12]:


#page title
st.set_page_config(page_title="Facial Emotion Detection")
#loading saved model
json_file=open('model_fer_4.json','r')
loaded_model_json=json_file.read()
json_file.close()
classifier= model_from_json(loaded_model_json)

classifier.load_weights("model_fer_4.h5")

#implimneting haarcascade
try:
    face_haar_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
except Exception:
    st.write("haarcascade loading error")

    #capturing face from camera stream
class VideoTranformer(VideoTransformerBase):
    def transform(self,frame):
        img=frame.to_ndarray(format="bgr24")
        
        grayscale=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces=face_haar_cascade.detectMultiScale(image=grayscale)
        
        #forming box around face
        for (x,y,w,h) in faces:
            cv2.rectangle(img=img, 
                          pt1=(x,y),
                         pt2=(x+w, y+h),
                         color=(255,0,0),
                         thickness=2)
            
            #converting obtained bgr image to grayscale
            roi_gray=grayscale[y-5:y+h+5,x-5:x+w+5] #adding buffer
            roi_gray=cv2.resize(roi_gray,(48,48))
            image_pixels=img_too_array(roi_array)
            image_pixels=np.expand_dims(image_pixels,axis=0)
            
            #normalizing
            
            image_pixels= image_pixels/255.
            predictions= classifier.predict(image_pixels)
            max_index=np.argmax(predictions[0])
            
            emotions= ['Angry','Disgusted','Fearful','Happy', 'Sad', 'Surprised', 'Neutral']
            emotion_prediction=emotions[max_index]
            
            font=cv2.FONT_HERSHEY_SIMPLEX
            lable_color=(0, 255, 0)
            cv2.putText(img,emotion_prediction, (int(x),int(y)), font,0.9,lable_color,2)
        return img


# In[13]:


def main():
    pages=["Home","About"]
    
    with st.sidebar :
        st.title('Select')
        
        page_name=st.selectbox('Select Page:',pages)
    st.title(page_name)
    
    
    if page_name==' Home':
        home_html=""" <body>
        <h4 style="font-size:30px"> Live Emotion Detection</h4>
        <p> This application detects emotion in real time from your camera feed using a CNN model. </p>
        </body>"""
        
        st.markdown(home_html, unsafe_allow_html=True)
        st.write("Click on START to begin")
        
        webrtc_streamer(key="example",video_transformer_factory=VideoTransformer, media_stream_constraints={
            "video":True,
            "audio":False},
        rtc_configuration=RTCConfiguration(
        {"iceServers":[{"urls":["stun:stun.1.google.com:19302"]}]}
        ) )
        
    elif page_name=="About":
        about_html="""<body>
        <h4 style="font-size:30px"> Facial emotion detection using CNN in real time</h4>
            <body>""" 
        
        
        st.markdown(about_html,unsafe_allow_html=True)
        st.write("This project is developed by Vivek CP, Data Science Trainee at AlmaBetter.")
        statement_html="""<body>
        <h4 style="font-size:20px"> About Project </h4>
        <p> Facial emotion recognition is an age old problem in the field of deep learning. The learning objective of the project is to gain hands on experience in developing a CNN model, and deploy it in real time. The model has achieved an training accuracy of 74% and test accuracy of 67%. The numbers can be improved by expanding the dataset and using techniques like transfer learning. This is included in the future scope of the project. <p>
        <body>"""
        
        st.markdown(statement_html,True)
        
    else:
        pass

if __name__=="__main__":
    main()


# In[ ]:





import streamlit as st
import cv2
import numpy as np
import tempfile
import requests
import time
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input

# =====================================
# CONFIG
# =====================================

CONFIDENCE_THRESHOLD = 0.4

TELEGRAM_BOT_TOKEN = "8045411577:AAEM2Q1t25zix1kpLKXIjYLIgstGI_zxYpM"
TELEGRAM_CHAT_ID = "5119209527" 

TELEGRAM_BOT_LINK = "https://t.me/Wildlife_animals_alert_systembot"

ALERT_COOLDOWN = 5

# =====================================
# SESSION
# =====================================

if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = 0

if "run" not in st.session_state:
    st.session_state.run = False

# =====================================
# CLASSES
# =====================================

class_names = [
    "Bear","Buffalo","Cheetah","Deer","Elephant",
    "Fox","Hyena","Jaguar","Leopard","Lion",
    "Monkey","Rhino","Tiger","Wolf","Zebra"
]

# =====================================
# LOAD MODELS
# =====================================

@st.cache_resource
def load_models():
    return YOLO("best.pt"), load_model("vgg19_model_final.keras", compile=False)

yolo_model, vgg_model = load_models()
_ = yolo_model(np.zeros((640,640,3), dtype=np.uint8))

# =====================================
# SMART FRAME CHECK
# =====================================

def is_frame_active(frame, prev_frame, threshold=25):

    if prev_frame is None:
        return True

    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    return np.count_nonzero(diff) > threshold * 100

# =====================================
# TELEGRAM ALERT
# =====================================

def send_telegram_alert(image_path, name, score, t):

    if time.time() - st.session_state.last_alert_time < ALERT_COOLDOWN:
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

        msg = f"""🚨 Wildlife Detected
Animal: {name}
Confidence: {score:.2f}
Prediction Time: {t:.3f} sec
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        with open(image_path, "rb") as img:
            requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": msg},
                files={"photo": img},
                timeout=5
            )

        st.session_state.last_alert_time = time.time()

    except Exception as e:
        print("Telegram Error:", e)

# =====================================
# PREDICTION
# =====================================

def hybrid_predict(frame):

    start = time.time()

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo_model(img_rgb, conf=0.25)[0]

    detections = []

    if results.boxes is None or len(results.boxes) == 0:
        return detections, 0.0

    h, w, _ = img_rgb.shape

    for box, conf, cls in zip(
        results.boxes.xyxy.cpu().numpy(),
        results.boxes.conf.cpu().numpy(),
        results.boxes.cls.cpu().numpy()
    ):

        x1,y1,x2,y2 = map(int, box)

        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(w,x2), min(h,y2)

        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        try:
            crop = cv2.resize(crop,(224,224))
            crop = np.expand_dims(crop,0)
            crop = preprocess_input(crop)

            vgg_score = float(np.max(vgg_model.predict(crop, verbose=0)))
        except:
            vgg_score = 0.0

        score = max(float(conf), vgg_score)

        detections.append((x1,y1,x2,y2,int(cls),score))

    return detections, time.time() - start

# =====================================
# UI
# =====================================

st.markdown("<h1 style='text-align:center;'>🐾 Wildlife Detection System</h1>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div style='text-align:center;'>
        <a href="{TELEGRAM_BOT_LINK}" target="_blank">
            <button style="background-color:#0088cc;color:white;padding:10px 20px;border:none;border-radius:8px;">
                📢 Join Telegram Bot
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

mode = st.radio("Select Input", ["Image","Video","Webcam"], horizontal=True)

# =====================================
# IMAGE
# =====================================

if mode == "Image":

    file = st.file_uploader("Upload Image")

    if file:
        frame = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_RGB2BGR)

        detections, t = hybrid_predict(frame)

        for x1,y1,x2,y2,label,score in detections:
            if score > CONFIDENCE_THRESHOLD:

                name = class_names[label]

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f"{name} {score:.2f}",
                            (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

                cv2.imwrite("img.jpg", frame)
                send_telegram_alert("img.jpg", name, score, t)

        cv2.putText(frame,f"{t:.3f}s",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        st.image(frame, channels="BGR")

# =====================================
# VIDEO
# =====================================

elif mode == "Video":

    file = st.file_uploader("Upload Video")

    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        prev_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if is_frame_active(frame, prev_frame):

                detections, t = hybrid_predict(frame)

                for x1,y1,x2,y2,label,score in detections:
                    if score > CONFIDENCE_THRESHOLD:

                        name = class_names[label]

                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(frame,f"{name} {score:.2f}",
                                    (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

                        cv2.imwrite("video.jpg", frame)
                        send_telegram_alert("video.jpg", name, score, t)

                cv2.putText(frame,f"{t:.3f}s",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            else:
                cv2.putText(frame,"No Activity",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

            stframe.image(frame, channels="BGR")
            prev_frame = frame.copy()

        cap.release()

# =====================================
# WEBCAM
# =====================================

elif mode == "Webcam":

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ Start"):
            st.session_state.run = True

    with col2:
        if st.button("⏹ Stop"):
            st.session_state.run = False

    if st.session_state.run:

        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        prev_frame = None

        while st.session_state.run:

            ret, frame = cap.read()
            if not ret:
                break

            if is_frame_active(frame, prev_frame):

                detections, t = hybrid_predict(frame)

                for x1,y1,x2,y2,label,score in detections:
                    if score > CONFIDENCE_THRESHOLD:

                        name = class_names[label]

                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(frame,f"{name} {score:.2f}",
                                    (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

                        cv2.imwrite("webcam.jpg", frame)
                        send_telegram_alert("webcam.jpg", name, score, t)

                cv2.putText(frame,f"{t:.3f}s",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            else:
                cv2.putText(frame,"No Activity",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

            stframe.image(frame, channels="BGR")
            prev_frame = frame.copy()

        cap.release()
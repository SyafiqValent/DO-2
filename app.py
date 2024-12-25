import os
from ultralytics import YOLO
import cv2
import streamlit as st
import time

# Load YOLO model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

def display_results(image, results):
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    labels = results.boxes.cls.cpu().numpy()
    names = results.names

    for i in range(len(boxes)):
        if scores[i] > 0.2:  # Confidence threshold
            x1, y1, x2, y2 = boxes[i].astype(int)
            label = names[int(labels[i])]
            score = scores[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Kotak hijau
            cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def main():
    st.title("Real-time Object Detection with YOLO")
    st.sidebar.title("Settings")

    model_path = r"D:\DO 2\runs\detect\train\weights\best.pt"
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan: {model_path}")
        return
    
    model = load_model(model_path)

    run_detection = st.sidebar.checkbox("Start/Stop Object Detection", key="detection_control")

    if run_detection:
        cap = cv2.VideoCapture(0)
        st_frame = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Gagal mengambil gambar.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(frame, imgsz=640, conf=0.2)

            if len(results[0].boxes) == 0:
                st.warning("Tidak ada objek yang terdeteksi.")
            else:
                frame = display_results(frame, results[0])
                st_frame.image(frame, channels="RGB", use_column_width=True)

            if not st.session_state.detection_control:
                break
            
            time.sleep(0.03)

        cap.release()

if __name__ == "__main__":
    main()

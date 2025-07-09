from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from collections import Counter

#load model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

st.title('Object Detection Application')
import tempfile

mode = st.radio("Choose detection mode:", ["Image", "Video"])

if mode == "Video":
    video_file = st.file_uploader("Upload a Video...", type=['mp4', 'mov', 'avi', 'mkv'])

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        stframe = st.empty()
        progress = st.progress(0)
        frame_text = st.empty()

        all_class_names = list(model.names.values())
        selected_classes = st.multiselect("Select classes to display", all_class_names, default=all_class_names)

        current_frame = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)[0]
            boxes = results.boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            class_names = [model.names[c] for c in class_ids]

            for box, cls_name, conf in zip(xyxy, class_names, confidences):
                if cls_name in selected_classes:
                    x1, y1, x2, y2 = box
                    label = f"{cls_name} {conf:.2f}"
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_rgb, label, (x1, y1 - 10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                                color=(255, 0, 0), thickness=1)

            out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            stframe.image(frame_rgb, channels="RGB", use_column_width=True)
            current_frame += 1
            progress.progress(min(current_frame / total_frames, 1.0))
            frame_text.text(f"Processing frame {current_frame}/{total_frames}")

        cap.release()
        out.release()

        st.success("✅ Video processing complete!")

        # download link
        with open(output_path, 'rb') as f:
            st.download_button(
                label="📥 Download Processed Video",
                data=f,
                file_name='detected_video.mp4',
                mime='video/mp4'
            )

if mode == "Image":
    upload = st.file_uploader('Upload an Image...', type = ['png', 'jpeg', 'jpg'])

    if upload is None:
        st.warning("Please upload an image file.")

    # In the Image mode section:
    if upload is not None:
        image = Image.open(upload).convert('RGB')  # Convert to RGB format
        image_array = np.array(image)
        with st.spinner("Detecting..."):
            results = model(image_array)[0]

        # results = model(image_array)[0]

        boxes = results.boxes

        class_ids = boxes.cls.cpu().numpy().astype(int)

        confidences = boxes.conf.cpu().numpy()

        xyxy = boxes.xyxy.cpu().numpy().astype(int)

        class_names = [model.names[c] for c in class_ids]

        unique_classes = sorted(set(class_names))

        selected_classes = st.multiselect('Image classes...', unique_classes, default= unique_classes)

        for box, cls_name, conf in zip(xyxy, class_names, confidences):
            if cls_name in selected_classes:
                x1, y1, x2, y2 = box
                label = f"{cls_name} {conf:.2f}"

                class_counter = Counter(class_names)
                count_text = ", ".join([f"{cls}: {cnt}" for cls, cnt in class_counter.items()])
                

                cv2.rectangle(image_array, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(image_array, label, (x1, y1 - 10),
                            fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale= 0.5, color= (255,0,0), thickness=1)
                

        st.write(f"Detected {len(class_names)} objects: {', '.join(unique_classes)}")

        st.image(image_array, use_column_width= True, caption= 'detected objects')
        st.info(f"Detected classes: {count_text}")
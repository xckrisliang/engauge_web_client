"""Video transforms with OpenCV"""

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import cvzone
import numpy as np

from sample_utils.turn import get_ice_servers
import os
import sys
from utils import detect_smile

draw_rect = True
cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')

overlay = cv2.imread('overlays/rayban2.png', cv2.IMREAD_UNCHANGED)
logo_overlay = cv2.imread('overlays/rayban_logo.png', cv2.IMREAD_UNCHANGED)
reward_overlay = cv2.imread('overlays/you_donated_a_smile.png', cv2.IMREAD_UNCHANGED)


_type = st.radio("Select transform type", ("noop", "cartoon", "edges", "rotate"))

count_filter_time = 0
smiling_time = 0
filter_time_threshold = 10
smile_time_threshold = 1 # how much time before the filter is shown
waiting_for_smile = False
length_smile_memory = 3 
fps = 30 #TODO: change 
smile_memory = []

def overlay_bgra(background: np.ndarray, overlay: np.ndarray, roi):
    roi_x, roi_y, roi_w, roi_h = roi
    roi_aspect_ratio = roi_w / roi_h

    # Calc overlay x, y, w, h that cover the ROI keeping the original aspect ratio
    ov_org_h, ov_org_w = overlay.shape[:2]
    ov_aspect_ratio = ov_org_w / ov_org_h

    if ov_aspect_ratio >= roi_aspect_ratio:
        ov_h = roi_h
        ov_w = int(ov_aspect_ratio * ov_h)
        ov_y = roi_y
        ov_x = int(roi_x - (ov_w - roi_w) / 2)
    else:
        ov_w = roi_w
        ov_h = int(ov_w / ov_aspect_ratio)
        ov_x = roi_x
        ov_y = int(roi_y - (ov_h - roi_h) / 2)

    resized_overlay = cv2.resize(overlay, (ov_w, ov_h))

    # Cut out the pixels of the overlay image outside the background frame.
    margin_x0 = -min(0, ov_x)
    margin_y0 = -min(0, ov_y)
    margin_x1 = max(background.shape[1], ov_x + ov_w) - background.shape[1]
    margin_y1 = max(background.shape[0], ov_y + ov_h) - background.shape[0]

    resized_overlay = resized_overlay[
        margin_y0 : resized_overlay.shape[0] - margin_y1,
        margin_x0 : resized_overlay.shape[1] - margin_x1,
    ]
    ov_x += margin_x0
    ov_w -= margin_x0 + margin_x1
    ov_y += margin_y0
    ov_h -= margin_y0 + margin_y1

    # Overlay
    foreground = resized_overlay[:, :, :3]
    mask = resized_overlay[:, :, 3]

    overlaid_area = background[ov_y : ov_y + ov_h, ov_x : ov_x + ov_w]
    overlaid_area[:] = np.where(mask[:, :, np.newaxis], foreground, overlaid_area)

def callback(frame: av.VideoFrame) -> av.VideoFrame:
    

    img = frame.to_ndarray(format="bgr24")
    if(_type == "noop"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = cascade_face.detectMultiScale(
            gray, scaleFactor=1.11, minNeighbors=3, minSize=(30, 30)
        )

        overlay = cv2.imread('overlays/rayban2.png', cv2.IMREAD_UNCHANGED)
        is_smiling = detect_smile(gray, faces, cascade_smile)

        
        for (x, y, w, h) in faces:
            # Ad-hoc adjustment of the ROI for each filter type
            roi = (x, y, w, h)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if is_smiling:
                smiling_time += 1
                if (smiling_time >= 20):
                    overlay_bgra(img, overlay, roi)
            else:
                smiling_time = 0
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    elif _type == "cartoon":
        # prepare color
        img_color = cv2.pyrDown(cv2.pyrDown(img))
        for _ in range(6):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        img_color = cv2.pyrUp(cv2.pyrUp(img_color))

        # prepare edges
        img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_edges = cv2.adaptiveThreshold(
            cv2.medianBlur(img_edges, 7),
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            2,
        )
        img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

        # combine color and edges
        img = cv2.bitwise_and(img_color, img_edges)
    elif _type == "edges":
        # perform edge detection
        img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    elif _type == "rotate":
        # rotate image
        rows, cols, _ = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
        img = cv2.warpAffine(img, M, (cols, rows))

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="opencv-filter",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    video_frame_callback=callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown(
    "This demo is based on "
    "https://github.com/aiortc/aiortc/blob/2362e6d1f0c730a0f8c387bbea76546775ad2fe8/examples/server/server.py#L34. "  # noqa: E501
    "Many thanks to the project."
)

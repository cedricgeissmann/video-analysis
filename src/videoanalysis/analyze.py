import cv2
import mediapipe as mp

import os
from pathlib import Path

from .landmarks import filters

mp_pose = mp.solutions.pose

class Globals:
    buffer = []
    def __init__(self, buffer_size=0):
        self.buffer_size = buffer_size


g = Globals(buffer_size=1)


def analyze(url: str, selected_filters: list, **kwargs):
    """
    Analyze a given video or directly from the webcam.

    :url: This is the url to the video that will be analyzed. If you want to
    analyze directly from the webcam, simply put 'life' as value for the url.

    :selected_filters: This is a list of predifened filter that you want to
    apply to the video. If the filter you are looking for does not exist,
    simple create a new one.
    """
    if url == "live":
        cap = cv2.VideoCapture(0)
        filename = "live.mp4"
    elif url == "webcam":
        cap = cv2.VideoCapture(kwargs["webcam"])
        filename = "webcam.mp4"
    else:
        cap = cv2.VideoCapture(url)
        filename = os.path.basename(url)
    pose = mp_pose.Pose()

    cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if not Path('analyzed').is_dir():
        Path('analyzed').mkdir()
    out = cv2.VideoWriter(os.path.join("analyzed", f'analyzed_{selected_filters}_{filename}'), cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if url == "live":
            frame = cv2.flip(frame, 1)

        if not url == "webcam":
            g.buffer.append(frame)
            if len(g.buffer) < g.buffer_size:
                continue
            frame = g.buffer.pop(0)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img)
        if res.pose_landmarks == None:
            continue

        # Draw landmarks accoring to the callback function provided.
        for filter_name, filt in filters.items():
            if filter_name in selected_filters:
                for f in filt:
                    f.apply(res, frame, frame_width, frame_height)

        cv2.imshow('cam', frame)
        out.write(frame)

        key_code = cv2.waitKey(1);
        if key_code & 0xFF == 27:
            break
        elif key_code & 0xFF != 255:
            handle_keys(key_code)
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def handle_keys(key_code):
    if key_code == ord('k'):
        g.buffer_size = min(g.buffer_size + 15, 30 * 10)
    elif key_code == ord('j'):
        g.buffer_size = max(g.buffer_size - 15, 1)
        while len(g.buffer) > g.buffer_size + 1:
            g.buffer.pop(0)

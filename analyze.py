import cv2
import mediapipe as mp

import os
from pathlib import Path

from landmarks import filters

mp_pose = mp.solutions.pose

def analyze(url: str, selected_filters: list):
    """
    Analyze a given video or directly from the webcam.

    :url: This is the url to the video that will be analyzed. If you want to
    analyze directly from the webcam, simply put 'life' as value for the url.

    :selected_filters: This is a list of predifened filter that you want to
    apply to the video. If the filter you are looking for does not exist,
    simple create a new one.
    """
    cap = cv2.VideoCapture(0) if url == "live" else cv2.VideoCapture(url)
    pose = mp_pose.Pose()

    filename = os.path.basename(url) if url != "live" else "live.mp4"

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

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

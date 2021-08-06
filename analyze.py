import cv2
import mediapipe as mp

import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

filters = {
    'box': {
        'landmarks': [
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE,
        ],
        'connections': [
            (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST),
            (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_ANKLE),
        ],
        'custom_landmarks': [(mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP)],
        'prolonged': []
        },
    'straight_arms': {
        'landmarks': [
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
        ],
        'connections': [
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        ],
        'custom_landmarks': [],
        'prolonged': []
        },
    'elbow_to_knee': {
        'landmarks': [
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.LEFT_KNEE,
        ],
        'connections': [
        ],
        'custom_landmarks': [
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_KNEE),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_KNEE),
            ],
        'prolonged': []
        },
    'setter': {
        'landmarks': [
        ],
        'connections': [
        ],
        'custom_landmarks': [],
        'prolonged': [
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, 3, 2),
            ]
        }
    }


def _get_pixel(lm, res, frame_width, frame_height):
    lm_0 = res.pose_landmarks.landmark[lm]
    return mp_drawing._normalized_to_pixel_coordinates(lm_0.x,
                                                       lm_0.y,
                                                       frame_width,
                                                       frame_height)

def _get_midpoint(px_0, px_1):
    return (
        int((px_0[0] + px_1[0]) / 2),
        int((px_0[1] + px_1[1]) / 2))


def _get_endpoints(px_0, px_1, length_1=3, length_2=1):
    direction = (px_0[0] - px_1[0], px_0[1] - px_1[1])
    return (
        (px_0[0] + length_1 * direction[0], px_0[1] + length_1 * direction[1]),
        (px_1[0] - length_2 * direction[0], px_1[1] - length_2 * direction[1])
        )


def analyze(url, filt):
    cap = cv2.VideoCapture(url)
    pose = mp_pose.Pose()

    filename = os.path.basename(url)

    cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(os.path.join("analyzed", f'analyzed_{filename}'), cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    active_filter = filters.get(filt)
    while True:
        ret, frame = cap.read()
        if not ret:
            break


        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img)
        if res.pose_landmarks == None:
            continue

        for point in active_filter['landmarks']:
            px = _get_pixel(point, res, frame_width, frame_height)
            cv2.circle(frame, px, 5, (255,0,0), -1)

        for con in active_filter['connections']:
            px_0 = _get_pixel(con[0], res, frame_width, frame_height)
            px_1 = _get_pixel(con[1], res, frame_width, frame_height)
            cv2.line(frame, px_0, px_1, (0,255,0), 5)


        for mid in active_filter['custom_landmarks']:
            px_0 = _get_pixel(mid[0], res, frame_width, frame_height)
            px_1 = _get_pixel(mid[1], res, frame_width, frame_height)

            if px_0 != None and px_1 != None:
                midpoint = _get_midpoint(px_0, px_1)
                cv2.circle(frame, midpoint, 10, (0,0,255), -1)
                endpoints = _get_endpoints(px_0, px_1)
                cv2.line(frame, endpoints[0], endpoints[1], (0,0,255), 5)

        for line in active_filter['prolonged']:
            px_0 = _get_pixel(line[0], res, frame_width, frame_height)
            px_1 = _get_pixel(line[1], res, frame_width, frame_height)

            if px_0 != None and px_1 != None:
                endpoints = _get_endpoints(px_0, px_1, line[2], line[3])
                cv2.line(frame, endpoints[0], endpoints[1], (0,0,255), 5)

        cv2.imshow('cam', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

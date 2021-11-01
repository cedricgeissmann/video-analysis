from abc import abstractmethod
import cv2

import utils

import mediapipe as mp
mp_pose = mp.solutions.pose

class Landmark():
    def __init__(self, landmarks, style={}):
        self.landmarks = landmarks
        self.style = style

    @abstractmethod
    def apply(self, res, frame, frame_width, frame_height):
        pass

class PointLandmark(Landmark):
    def apply(self, res, frame, frame_width, frame_height):
        for point in self.landmarks:
            px = utils._get_pixel(point, res, frame_width, frame_height)
            cv2.circle(frame, px, 5, (255,0,0), -1)


class ConnectionLandmark(Landmark):
    def __init__(self, landmarks, style={}):
        assert(len(landmarks) > 0)
        assert(len(landmarks[0]) == 2)
        Landmark.__init__(self, landmarks, style)

    def apply(self, res, frame, frame_width, frame_height):
        for con in self.landmarks:
            px_0 = utils._get_pixel(con[0], res, frame_width, frame_height)
            px_1 = utils._get_pixel(con[1], res, frame_width, frame_height)
            cv2.line(frame, px_0, px_1, self.style['color'] or (255,0,0), 5)


class MidpointLandmark(Landmark):
    def apply(self, res, frame, frame_width, frame_height):
        for mid in self.landmarks:
            px_0 = utils._get_pixel(mid[0], res, frame_width, frame_height)
            px_1 = utils._get_pixel(mid[1], res, frame_width, frame_height)

            if px_0 != None and px_1 != None:
                midpoint = utils._get_midpoint(px_0, px_1)
                cv2.circle(frame, midpoint, 10, (0,0,255), -1)


class ProlongedLandmark(Landmark):
    def apply(self, res, frame, frame_width, frame_height):
        for line in self.landmarks:
            px_0 = utils._get_pixel(line[0], res, frame_width, frame_height)
            px_1 = utils._get_pixel(line[1], res, frame_width, frame_height)

            if px_0 != None and px_1 != None:
                endpoints = utils._get_endpoints(px_0, px_1, line[2], line[3])
                cv2.line(frame, endpoints[0], endpoints[1], (0,0,255), 5)


filters = {
        'test': [
            MidpointLandmark(
                landmarks=[(mp_pose.PoseLandmark.LEFT_WRIST,
                    mp_pose.PoseLandmark.RIGHT_WRIST)]
                ),
            ConnectionLandmark(
                landmarks=[(mp_pose.PoseLandmark.LEFT_WRIST,
                    mp_pose.PoseLandmark.RIGHT_WRIST)],
                style={'color': (255, 255, 0)}
                )
            ],
        'box': [
            PointLandmark(
                landmarks=[
                    mp_pose.PoseLandmark.LEFT_WRIST,
                    mp_pose.PoseLandmark.RIGHT_WRIST,
                    mp_pose.PoseLandmark.LEFT_ANKLE,
                    mp_pose.PoseLandmark.RIGHT_ANKLE,
                    ]
                ),
            ConnectionLandmark(
                landmarks=[
                    (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST),
                    (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_ANKLE),
                    (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE),
                    (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_ANKLE),
                    ]
                ),
            MidpointLandmark(
                landmarks=[
                    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP)
                    ]
                )
            ],
        'straight_arms': [
            PointLandmark(
                landmarks=[
                    mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.RIGHT_ELBOW,
                    mp_pose.PoseLandmark.RIGHT_WRIST,
                    mp_pose.PoseLandmark.LEFT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_ELBOW,
                    mp_pose.PoseLandmark.LEFT_WRIST,
                    ]
                ),
            ConnectionLandmark(
                landmarks=[
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                    ]
                )
            ],
        'setter': [
                ProlongedLandmark(
                    landmarks=[
                        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, 3, 2),
                        ]
                    )
                ]
        }

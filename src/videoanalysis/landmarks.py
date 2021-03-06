from abc import abstractmethod
from math import acos, degrees, sqrt
import cv2

from . import utils

import mediapipe as mp
mp_pose = mp.solutions.pose


default_style = {
    'color': (255, 0, 0),
    'line-color': (0, 255, 0),
    'dot-radius': 20,
    'line-width': 5
}

class Landmark():
    def __init__(self, landmarks, style={}):
        self.landmarks = landmarks
        self.style = default_style.copy()
        for key, val in style.items():
            self.style[key] = val

    @abstractmethod
    def apply(self, res, frame, frame_width, frame_height):
        pass

class PointLandmark(Landmark):
    def apply(self, res, frame, frame_width, frame_height):
        for point in self.landmarks:
            px = utils._get_pixel(point, res, frame_width, frame_height)
            cv2.circle(frame, px,
                self.style['dot-radius'],
                self.style['color'],
                -1
            )


class ConnectionLandmark(Landmark):
    def __init__(self, landmarks, style={}):
        assert(len(landmarks) > 0)
        Landmark.__init__(self, landmarks, style)

    def apply(self, res, frame, frame_width, frame_height):
        for con in self.landmarks:
            if len(con) == 2:
                px_0 = utils._get_pixel(con[0], res, frame_width, frame_height)
                px_1 = utils._get_pixel(con[1], res, frame_width, frame_height)
                cv2.line(frame, px_0, px_1,
                    self.style['line-color'],
                    self.style['line-width']
                )
            elif len(con) == 4:
                px_0 = utils._get_pixel(con[0], res, frame_width, frame_height)
                px_1 = utils._get_pixel(con[1], res, frame_width, frame_height)
                px_2 = utils._get_pixel(con[2], res, frame_width, frame_height)
                px_3 = utils._get_pixel(con[3], res, frame_width, frame_height)

                px_1 = utils._get_midpoint(px_1, px_2)
                cv2.line(frame, px_0, px_1,
                    self.style['line-color'],
                    self.style['line-width']
                )
                cv2.line(frame, px_1, px_3,
                    self.style['line-color'],
                    self.style['line-width']
                )


class ClosePoints(Landmark):
    def apply(self, res, frame, frame_width, frame_height):
        for lms in self.landmarks:
            p0 = res.pose_world_landmarks.landmark[lms[0]]
            p1 = res.pose_world_landmarks.landmark[lms[1]]

            # Project onto xy-plane
            v0 = (p0.x, p0.y)
            v1 = (p1.x, p1.y)

            dist = sqrt((v0[0] - v1[0])**2 + (v0[1] - v1[1])**2)

            print(dist)
            color = (0, 255, 0) if dist < 0.18 else (0, 0, 255)

            px = utils._get_pixel(lms[0], res, frame_width, frame_height)
            cv2.circle(frame, px,
                self.style['dot-radius'],
                color,
                -1
            )


class MidpointLandmark(Landmark):
    def apply(self, res, frame, frame_width, frame_height):
        for mid in self.landmarks:
            px_0 = utils._get_pixel(mid[0], res, frame_width, frame_height)
            px_1 = utils._get_pixel(mid[1], res, frame_width, frame_height)

            if px_0 != None and px_1 != None:
                midpoint = utils._get_midpoint(px_0, px_1)
                cv2.circle(frame, midpoint,
                    self.style['dot-radius'],
                    self.style['color'],
                    -1
                )


class ProlongedLandmark(Landmark):
    def __init__(self, landmarks, style={}, lengthen_first=0, lengthen_second=0):
        super().__init__(landmarks, style=style)
        self.lengthen_first = lengthen_first
        self.lengthen_second = lengthen_second

    def apply(self, res, frame, frame_width, frame_height):
        for line in self.landmarks:
            px_0 = utils._get_pixel(line[0], res, frame_width, frame_height)
            px_1 = utils._get_pixel(line[1], res, frame_width, frame_height)

            if px_0 != None and px_1 != None:
                endpoints = utils._get_endpoints(px_0, px_1, self.lengthen_first, self.lengthen_second)
                cv2.line(frame, endpoints[0], endpoints[1],
                    self.style['line-color'],
                    self.style['line-width']
                )


class AngleLandmark(Landmark):
    def apply(self, res, frame, frame_width, frame_height):
        for lms in self.landmarks:
            p0 = res.pose_world_landmarks.landmark[lms[0]]
            p1 = res.pose_world_landmarks.landmark[lms[1]]
            p2 = res.pose_world_landmarks.landmark[lms[2]]

            v1 = (p0.x - p1.x, p0.y - p1.y, p0.z - p1.z)
            v2 = (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)

            v1 = utils._normalize(v1)
            v2 = utils._normalize(v2)

            dotp = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

            angle = degrees(acos(dotp))

            line_color = (0, 255, 0) if 150 < angle < 210 else (0, 0, 255)

            clm = ConnectionLandmark(
                landmarks=[(lms[0], lms[1]), (lms[1], lms[2])],
                style={'line-color': line_color}
            )
            clm.apply(res, frame, frame_width, frame_height)


class AngleHIPLandmark(Landmark):
    def apply(self, res, frame, frame_width, frame_height):
        for lms in self.landmarks:
            p0 = res.pose_world_landmarks.landmark[lms[0]]
            p1 = res.pose_world_landmarks.landmark[lms[1]]
            p2 = res.pose_world_landmarks.landmark[lms[2]]
            p3 = res.pose_world_landmarks.landmark[lms[3]]

            m = ((p1.x + p2.x)/2, (p1.y + p2.y)/2, (p1.z + p2.z)/2)

            v1 = (p0.x - m[0], p0.y - m[1], p0.z - m[2])
            v2 = (p3.x - m[0], p3.y - m[1], p3.z - m[2])

            v1 = utils._normalize(v1)
            v2 = utils._normalize(v2)

            dotp = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

            angle = degrees(acos(dotp))

            line_color = (0, 255, 0) if 170 < angle < 190 else (0, 0, 255)

            clm = ConnectionLandmark(
                landmarks=[(lms[0], lms[1], lms[2], lms[3])],
                style={'line-color': line_color}
            )
            clm.apply(res, frame, frame_width, frame_height)


class ProlongedMidpointsLandmark(Landmark):
    def __init__(self, landmarks, style={}, lengthen_first=0, lengthen_second=0):
        super().__init__(landmarks, style=style)
        self.lengthen_first = lengthen_first
        self.lengthen_second = lengthen_second

    def apply(self, res, frame, frame_width, frame_height):
        for line in self.landmarks:

            px_0 = utils._get_pixel(line[0], res, frame_width, frame_height)
            px_1 = utils._get_pixel(line[1], res, frame_width, frame_height)
            px_2 = utils._get_pixel(line[2], res, frame_width, frame_height)
            px_3 = utils._get_pixel(line[3], res, frame_width, frame_height)

            if px_0 != None and px_1 != None and px_2 != None and px_3 != None:
                m1 = utils._get_midpoint(px_0, px_1)
                m2 = utils._get_midpoint(px_2, px_3)

                endpoints = utils._get_endpoints(m1, m2, self.lengthen_first, self.lengthen_second)
                cv2.line(frame, endpoints[0], endpoints[1],
                    self.style['line-color'],
                    self.style['line-width']
                )

filters = {
        '3d': [
            AngleLandmark(
                landmarks=[(
                    mp_pose.PoseLandmark.LEFT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_ELBOW,
                    mp_pose.PoseLandmark.LEFT_WRIST,
                    ), (
                        mp_pose.PoseLandmark.RIGHT_SHOULDER,
                        mp_pose.PoseLandmark.RIGHT_ELBOW,
                        mp_pose.PoseLandmark.RIGHT_WRIST,
                        )]
                    ),
            PointLandmark(
                landmarks=[
                    mp_pose.PoseLandmark.LEFT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_ELBOW,
                    mp_pose.PoseLandmark.LEFT_WRIST,
                    ]
                )
            ],
        'test': [
            ProlongedLandmark(
                landmarks=[(mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST)],
                lengthen_first=200,
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
                    ],
                style={'color': (255, 255, 0)}
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
                    landmarks=[(mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP)],
                    lengthen_second=200,
                    )
                ],
        'middle_axis': [
                ProlongedMidpointsLandmark(
                    landmarks=[
                        (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                            mp_pose.PoseLandmark.LEFT_SHOULDER,
                            mp_pose.PoseLandmark.RIGHT_HIP,
                            mp_pose.PoseLandmark.LEFT_HIP),
                        ],
                    lengthen_second=200,
                    style={'line-color': (255,0,255)}
                    )
                ],

    'Badminton': [
            AngleLandmark(
                landmarks=[(
                    mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.RIGHT_ELBOW,
                    mp_pose.PoseLandmark.RIGHT_WRIST,
                    )],) ,

                PointLandmark(
                    landmarks=[
                        mp_pose.PoseLandmark.RIGHT_WRIST,
                        ]
                    ) ,
                ProlongedMidpointsLandmark(
                    landmarks=[
                        (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                            mp_pose.PoseLandmark.LEFT_SHOULDER,
                            mp_pose.PoseLandmark.RIGHT_HIP,
                            mp_pose.PoseLandmark.LEFT_HIP),
                        ],
                    lengthen_second=200 ,
                    style={'color': (255, 0, 255)}
                    )
                ],

    'Fuss_Huefte_Schulter_Winkel': [
            AngleHIPLandmark(
                landmarks=[(
                    mp_pose.PoseLandmark.RIGHT_WRIST,
                    mp_pose.PoseLandmark.RIGHT_HIP,
                    mp_pose.PoseLandmark.LEFT_HIP,
                    mp_pose.PoseLandmark.LEFT_ANKLE,
                    )]
                ),
            ],
    'Handstand3': [
            AngleLandmark(
                landmarks=[(
                    mp_pose.PoseLandmark.LEFT_ANKLE,
                    mp_pose.PoseLandmark.LEFT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_WRIST,
                    ), ]
                ),
            PointLandmark(
                landmarks=[
                    mp_pose.PoseLandmark.LEFT_ANKLE,
                    mp_pose.PoseLandmark.LEFT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_WRIST,
                    ]
                )],

'Hdst4': [
        AngleLandmark(
            landmarks=[(
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_WRIST,

                ), (mp_pose.PoseLandmark.LEFT_ANKLE,
                    mp_pose.PoseLandmark.LEFT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_WRIST,
                    ) ]
                ),
        PointLandmark(
            landmarks=[
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.LEFT_ANKLE
                ]
            )],

        'Hilfslinie': [
                ProlongedLandmark(
                    landmarks=[(mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_WRIST)],
                    lengthen_second=300,
                    style={
                        "line-color": (0, 200, 200),
                        "line-width": 2
                        }
                    ),
                ConnectionLandmark(
                    landmarks=[
                        (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_ANKLE),
                        ]
                    )
                ],
        'close': [
                ClosePoints(landmarks=[(
                    mp_pose.PoseLandmark.RIGHT_WRIST,
                    mp_pose.PoseLandmark.LEFT_WRIST,
                    )])
                ],
        'con': [
                ConnectionLandmark(landmarks=[(
                    mp_pose.PoseLandmark.RIGHT_ANKLE,
                    mp_pose.PoseLandmark.RIGHT_HIP,
                    mp_pose.PoseLandmark.LEFT_HIP,
                    mp_pose.PoseLandmark.LEFT_WRIST,
                    )])
                ],

        'middle_axis2': [
                ProlongedMidpointsLandmark(
                    landmarks=[
                        (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                            mp_pose.PoseLandmark.LEFT_SHOULDER,
                            mp_pose.PoseLandmark.RIGHT_HIP,
                            mp_pose.PoseLandmark.LEFT_HIP),
                        ],
                    lengthen_second=200
                    ),

                AngleLandmark(
                    landmarks=[

                        (mp_pose.PoseLandmark.LEFT_SHOULDER,
                            mp_pose.PoseLandmark.LEFT_ELBOW,
                            mp_pose.PoseLandmark.LEFT_WRIST,
                            )]
                        ),

                ],

        'middle_axis3': [
                ProlongedMidpointsLandmark(
                    landmarks=[
                        (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                            mp_pose.PoseLandmark.RIGHT_ANKLE,
                            mp_pose.PoseLandmark.RIGHT_HIP,
                            mp_pose.PoseLandmark.LEFT_HIP),
                        ],
                    lengthen_second=200
                    ),

                AngleLandmark(
                    landmarks=[

                        (mp_pose.PoseLandmark.LEFT_SHOULDER,
                            mp_pose.PoseLandmark.LEFT_ELBOW,
                            mp_pose.PoseLandmark.LEFT_WRIST,
                            )]
                        ),

                ],

        'Gestreckter Arm Verl??ngerung rechts Tennis': [

                PointLandmark(
                    landmarks=[
                        mp_pose.PoseLandmark.LEFT_SHOULDER,
                        mp_pose.PoseLandmark.LEFT_ELBOW,
                        mp_pose.PoseLandmark.LEFT_WRIST,
                        ]
                    ),
                ProlongedLandmark(
                    landmarks=[
                        (mp_pose.PoseLandmark.LEFT_SHOULDER,
                            mp_pose.PoseLandmark.LEFT_ELBOW,
                            mp_pose.PoseLandmark.LEFT_WRIST),
                        ],
                    lengthen_first=300,
                    style = {
                        'line-color': (153, 255, 255),
                        'line-width': 2
                        },
                    ),

                ConnectionLandmark(
                    landmarks=[
                        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                        ],

                    ),

                AngleLandmark (
                    landmarks=[

                        (mp_pose.PoseLandmark.LEFT_SHOULDER,
                            mp_pose.PoseLandmark.LEFT_ELBOW,
                            mp_pose.PoseLandmark.LEFT_WRIST),
                        ],
                    )],

            'Gestreckter Arm Verl??ngerung rechts Badminton': [

                    PointLandmark(
                        landmarks=[
                            mp_pose.PoseLandmark.LEFT_SHOULDER,
                            mp_pose.PoseLandmark.LEFT_ELBOW,
                            mp_pose.PoseLandmark.LEFT_WRIST,
                            ]
                        ),
                    ProlongedLandmark(
                        landmarks=[
                            (mp_pose.PoseLandmark.LEFT_SHOULDER,
                                mp_pose.PoseLandmark.LEFT_ELBOW,
                                mp_pose.PoseLandmark.LEFT_WRIST),
                            ],
                        lengthen_first=250,
                        style = {
                            'line-color': (153, 255, 255),
                            'line-width': 2
                            },
                        ),

                    ConnectionLandmark(
                        landmarks=[
                            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                            ],

                        ),

                    AngleLandmark (
                        landmarks=[

                            (mp_pose.PoseLandmark.LEFT_SHOULDER,
                                mp_pose.PoseLandmark.LEFT_ELBOW,
                                mp_pose.PoseLandmark.LEFT_WRIST),
                            ],
                        )],

            'Gestreckter Arm Verl??ngerung links Tennis': [

                    PointLandmark(
                        landmarks=[
                            mp_pose.PoseLandmark.RIGHT_SHOULDER,
                            mp_pose.PoseLandmark.RIGHT_ELBOW,
                            mp_pose.PoseLandmark.RIGHT_WRIST,
                            ]
                        ),
                    ProlongedLandmark(
                        landmarks=[
                            (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                mp_pose.PoseLandmark.RIGHT_ELBOW,
                                mp_pose.PoseLandmark.RIGHT_WRIST),
                            ],
                        lengthen_first=300,
                        style = {
                            'line-color': (153, 255, 255),
                            'line-width': 2
                            },
                        ),

                    ConnectionLandmark(
                        landmarks=[
                            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                            ],

                        ),

                    AngleLandmark (
                        landmarks=[

                            (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                mp_pose.PoseLandmark.RIGHT_ELBOW,
                                mp_pose.PoseLandmark.RIGHT_WRIST),
                            ],
                        )],

'Gestreckter Arm Verl??ngerung links Badminton': [

        PointLandmark(
            landmarks=[
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_WRIST,
                ]
            ),
        ProlongedLandmark(
            landmarks=[
                (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.RIGHT_ELBOW,
                    mp_pose.PoseLandmark.RIGHT_WRIST),
                ],
            lengthen_first=250,
            style = {
                'line-color': (153, 255, 255),
                'line-width': 2
                },
            ),

        ConnectionLandmark(
            landmarks=[
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                ],

            ),

        AngleLandmark (
            landmarks=[

                (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.RIGHT_ELBOW,
                    mp_pose.PoseLandmark.RIGHT_WRIST),
                ],
            )],

        'empty': [
                ProlongedMidpointsLandmark(
                    landmarks=[
                        (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                            mp_pose.PoseLandmark.LEFT_SHOULDER,
                            mp_pose.PoseLandmark.RIGHT_HIP,
                            mp_pose.PoseLandmark.LEFT_HIP),
                        ],
                    lengthen_second=200
                    ),

                AngleLandmark(
                    landmarks=[
                        (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                            mp_pose.PoseLandmark.RIGHT_ELBOW,
                            mp_pose.PoseLandmark.RIGHT_WRIST,),

                        (mp_pose.PoseLandmark.LEFT_SHOULDER,
                            mp_pose.PoseLandmark.LEFT_ELBOW,
                            mp_pose.PoseLandmark.LEFT_WRIST,
                            )]
                        ),
                PointLandmark(
                    landmarks=[
                        mp_pose.PoseLandmark.LEFT_SHOULDER,
                        mp_pose.PoseLandmark.LEFT_ELBOW,
                        mp_pose.PoseLandmark.LEFT_WRIST,
                        mp_pose.PoseLandmark.RIGHT_SHOULDER,
                        mp_pose.PoseLandmark.RIGHT_ELBOW,
                        mp_pose.PoseLandmark.RIGHT_WRIST,
                        ]
                    )
                ]
        }

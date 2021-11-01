import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

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



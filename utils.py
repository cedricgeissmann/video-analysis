import mediapipe as mp
from math import sqrt

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
    """
    Calculate the endpoints of a line over the points defining them.

    This method takes 2 points that define a line. Then a vector is calculated
    from point 1 to point 2, which is the direction of the line.
    Then length_1 is appended to the first point, and length_2 is appended to
    the second point.

    Since the vector gets normalized, length_1 and length_2 can be estimated in
    pixels.
    """
    direction = (px_1[0] - px_0[0], px_1[1] - px_0[1])
    direction = _normalize(direction)
    return (
        (int(px_1[0] + length_1 * direction[0]), int(px_1[1] + length_1 * direction[1])),
        (int(px_0[0] - length_2 * direction[0]), int(px_0[1] - length_2 * direction[1]))
        )


def _normalize(vec):
    assert(len(vec) == 2)
    magnitude = sqrt(vec[0]**2 + vec[1]**2)
    return (vec[0] / magnitude, vec[1] / magnitude)



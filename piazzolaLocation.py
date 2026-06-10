import numpy as np


def getMedianGradient(lines):
    gradients = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            angleRad = np.arctan2(y2 - y1, x2 - x1)

            if angleRad < 0:
                angleRad += np.pi

            gradients.append(angleRad)

    if gradients:
        medianAngleRad = np.median(gradients)
        medianAngleDeg = np.degrees(medianAngleRad)
        return medianAngleDeg
    else:
        return None


def classifyPoint(pointX, pointY, angleDeg):
    x0, y0 = 112, 112
    angleRad = np.radians(angleDeg)

    D = np.cos(angleRad) * (pointY - y0) - np.sin(angleRad) * (pointX - x0)

    if D > 0:
        return 1
    else:
        return -1


def roadAngleAt(lineGeom, distance):
    eps = 1e-6
    d1 = max(0, distance - eps)
    d2 = min(lineGeom.length, distance + eps)
    p1 = lineGeom.interpolate(d1)
    p2 = lineGeom.interpolate(d2)

    dx = p2.x - p1.x
    dy = p2.y - p1.y
    angle = np.degrees(np.arctan2(dx, dy)) % 360
    return angle


def pixelGradientToAngle(gradientDeg):
    return (90 + gradientDeg) % 360


def getRoadSide(location, pixelGradientAngle, roadAngle):
    diff = abs(pixelGradientAngle - roadAngle) % 360

    flipped = 90 < diff < 270

    if flipped:
        location = -location

    return "left" if location == 1 else "right"

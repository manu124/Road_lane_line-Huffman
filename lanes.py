import cv2
import numpy as np




# optimization process\


def make_coordinate(image1, line_parameter):
    slope, intercept = line_parameter
    y1 = image1.shape[0]

    y2 = int(y1 * (3 / 5))  # line will start from below and then go to 3/5 th of picture

    x1 = int((y1 - intercept) / slope)  # as we know y = mx+ c soo x = (y-c)/m
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameter = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameter[0]
        intersept = parameter[1]
        if slope < 0:
            left_fit.append((slope, intersept))
        else:
            right_fit.append((slope, intersept))
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line = make_coordinate(image, left_fit_avg)
    right_line = make_coordinate(image, right_fit_avg)
    return np.array([left_line, right_line])


# before optimize

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):
    lines_image = np.zeros_like(image)  # black image
    if lines is not None:
        for line1 in lines:
            x1, y1, x2, y2 = line1.reshape(4)
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image


def region_of_interest(image):
    height = image.shape[0]
    # we have created a array of polygon &single
    polygon = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)  # it will creat a full screen with black(0)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# image = cv2.imread('test_image.jpg')

# frame_vedio = np.copy(image)  # if we use = image without copy change will reflect
# canny_image = canny(frame_vedio)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# average_line = average_slope(frame_vedio, lines)
# line_image = display_lines(frame_vedio, average_line)
# join_image = cv2.addWeighted(frame_vedio, 0.8, line_image, 1, 1)
# cv2.imshow("result", join_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture("vedio.mp4")
while (cap.isOpened()):
    _, frame_vedio = cap.read()
    canny_image = canny(frame_vedio)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    average_line = average_slope(frame_vedio, lines)
    line_image = display_lines(frame_vedio, average_line)
    join_image = cv2.addWeighted(frame_vedio, 0.8, line_image, 1, 1)
    cv2.imshow("result", join_image)
    if cv2.waitKey(1) == ord('q'):#0xFF(mak int to 8 bit) ==ord('q)
        break
cap.release()
cv2.destroyAllWindows()



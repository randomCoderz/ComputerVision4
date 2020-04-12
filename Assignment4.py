import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    # Convert the image to gray-scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image
    blur_img = cv2.GaussianBlur(gray, (5, 5), 0)

    # Find the edges in the image using canny detector
    get_edge = cv2.Canny(blur_img, 10, 40, apertureSize=3)
    return get_edge


# Create mask to get area of interest
def region_of_interest(image):
    mask = np.zeros_like(image)
    # Create a rectangle mask to cover pool table
    rectangle = np.array([[850, 520], [1425, 550], [1350, 1020], [110, 905]])
    cv2.fillPoly(mask, [rectangle], 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def region_of_interest_corner(image):
    corner_mask = np.zeros_like(image)
    # Create a rectangle mask to cover pool table
    area = np.array([[850, 500], [1460, 530], [1350, 1100], [50, 870]])
    cv2.fillPoly(corner_mask, [area], 255)
    masked_image = cv2.bitwise_and(image, corner_mask)
    return masked_image


if __name__ == '__main__':
    # Reading the image
    img = cv2.imread('table.jpg')
    edges = canny(img)
    roi = region_of_interest(edges)
    dilation = cv2.dilate(roi, (5, 5), iterations=1)
    cv2.imwrite('test.jpg', dilation)
    # Image, rho, theta, threshold
    lines = cv2.HoughLinesP(dilation, 2, np.pi / 180, 340, minLineLength=450, maxLineGap=200)

    x = 0
    line_image = img.copy()
    if lines is not None:
        for line in lines:
            if x == 1 or x == 2 or x == 4 or x == 9:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            x += 1

    cv2.imwrite("linesDetected.jpg", line_image)

    # Circle Image
    circle_image = img.copy()
    gray_circle = cv2.cvtColor(circle_image, cv2.COLOR_BGR2GRAY)
    circle_roi = region_of_interest(gray_circle)
    # 50, 30
    circles = cv2.HoughCircles(circle_roi, cv2.HOUGH_GRADIENT, 1, 20, param1=95, param2=25, minRadius=7, maxRadius=180)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # corresponding to the center of the circle
            cv2.circle(circle_image, (x, y), r, (0, 255, 0), 4)

    cv2.imwrite("circle.jpg", circle_image)

    # Corner Detection
    corner_image = img.copy()
    corner_edges = canny(corner_image)
    roi_corner = region_of_interest_corner(corner_edges)

    final_img = np.float32(roi_corner)
    dst = cv2.cornerHarris(final_img, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    corner_image[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imwrite('cornerHarris.jpg', corner_image)

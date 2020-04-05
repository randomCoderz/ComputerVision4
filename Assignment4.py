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
    # rectangle = np.array([[50, 500], [1500, 500], [1500, 1300], [50, 1300]])
    rectangle = np.array([[800, 500], [1450, 500], [1500, 1150], [0, 900]])
    cv2.fillPoly(mask, [rectangle], 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


if __name__ == '__main__':
    # Reading the image
    img = cv2.imread('table.jpg')
    edges = canny(img)
    plt.imshow(edges)
    roi = region_of_interest(edges)
    cv2.imwrite('test.jpg', roi)

    # Image, rho, theta, threshold
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, minLineLength=10, maxLineGap=50)
    #
    # for x1, y1, x2, y2 in lines[0]:
    #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #
    # cv2.imwrite("linesDetected.jpg", img)

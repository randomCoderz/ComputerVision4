import cv2
import numpy as np


# Find the Gaussian pyramid of the two images and the mask
def gaussian_pyramid(img, levels):
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(np.float32(lower))
    return gaussian_pyr


# Then calculate the Laplacian pyramid
def laplacian_pyramid(gaussian_pyr):
    laplacian_top = gaussian_pyr[-1]
    levels = len(gaussian_pyr) - 1

    laplacian_pyr = [laplacian_top]
    for i in range(levels, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = np.subtract(gaussian_pyr[i - 1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr


# Now blend the two images wrt. the mask
def blend(laplacian_A, laplacian_B, mask_pyr):
    LS = []
    for la, lb, mask in zip(laplacian_A, laplacian_B, mask_pyr):
        ls = lb * mask + la * (1.0 - mask)
        LS.append(ls)
    return LS


# Reconstruct the original image
def reconstruct(laplacian_pyr):
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    levels = len(laplacian_pyr) - 1
    for i in range(levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(laplacian_pyr[i + 1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)
    return laplacian_lst


# Now let's call all these functions
if __name__ == '__main__':
    # Step-1
    # Load the two images
    img1 = cv2.imread('img1.jpg').astype('float32')
    img2 = cv2.imread('img2.jpg').astype('float32')

    # Applying mask into binary image and then converting it to color
    mask = cv2.imread('img3.jpg', 0)
    mask = cv2.bilateralFilter(mask, 9, 75, 75)
    # Applying threshold to get true binary image
    # ret is the optimal threshold value for using Otsu's thresholding
    ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask = mask.astype('float32') / 255

    lvls = 7

    # For image-1, calculate Gaussian and Laplacian
    gaussian_pyr_1 = gaussian_pyramid(img1, lvls)
    laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)
    # For image-2, calculate Gaussian and Laplacian
    gaussian_pyr_2 = gaussian_pyramid(img2, lvls)
    laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)
    # Calculate the Gaussian pyramid for the mask image and reverse it.
    mask_pyr_final = gaussian_pyramid(mask, lvls)
    mask_pyr_final.reverse()
    # Blend the images
    add_laplace = blend(laplacian_pyr_1, laplacian_pyr_2, mask_pyr_final)
    # Reconstruct the images
    final = reconstruct(add_laplace)

    # Save the final image to the disk
    cv2.imwrite("final.jpg", final[lvls])

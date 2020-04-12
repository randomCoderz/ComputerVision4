import cv2
import numpy as np


# Find the Gaussian pyramid of the two images and the mask
def gauss_pyr(img, levels):
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(np.float32(lower))
    return gaussian_pyr


# Then calculate the Laplacian pyramid
def laplacian_pyramid(gaussian_pyr):
    top_lapl = gaussian_pyr[-1]
    levels = len(gaussian_pyr) - 1

    laplacian_pyr = [top_lapl]
    for i in range(levels, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = np.subtract(gaussian_pyr[i - 1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr


# Now blend the two images wrt. the mask
def blend(lapl_one, lapl_two, mask_pyr, mask_one):
    blend_var = []
    for la, lb, mask_one in zip(lapl_one, lapl_two, mask_pyr):
        ls = lb * mask_one + la * (1.0 - mask_one)
        blend_var.append(ls)
    return blend_var


# Reconstruct the original image
def reconstruct(laplacian_pyr):
    top_lapl = laplacian_pyr[0]
    laplacian_lst = [top_lapl]
    levels = len(laplacian_pyr) - 1
    for i in range(levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(top_lapl, dstsize=size)
        top_lapl = cv2.add(laplacian_pyr[i + 1], laplacian_expanded)
        laplacian_lst.append(top_lapl)
    return laplacian_lst


# Now let's call all these functions
if __name__ == '__main__':
    # Step-1
    # Load the two images
    img1 = cv2.imread('img1.jpg').astype('float32')
    img2 = cv2.imread('img2.jpg').astype('float32')

    # Applying mask into binary image and then converting it to color
    mask = cv2.imread('img3.jpg', 0)
    mask = cv2.GaussianBlur(mask, (13, 13), 1)
    ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask = mask.astype('float32') / 255

    lvls = 5

    # For image-1, calculate Gaussian and Laplacian
    g_pyr1 = gauss_pyr(img1, lvls)
    lapl_pyr1 = laplacian_pyramid(g_pyr1)
    # For image-2, calculate Gaussian and Laplacian
    g_pyr2 = gauss_pyr(img2, lvls)
    lapl_pyr2 = laplacian_pyramid(g_pyr2)
    # Calculate the Gaussian pyramid for the mask image and reverse it.
    pyr_mask = gauss_pyr(mask, lvls)
    pyr_mask.reverse()
    # Blend the images
    add_laplace = blend(lapl_pyr1, lapl_pyr2, pyr_mask, mask)
    # Reconstruct the images
    final = reconstruct(add_laplace)

    # Save the final image to the disk
    cv2.imwrite("final.jpg", final[lvls])

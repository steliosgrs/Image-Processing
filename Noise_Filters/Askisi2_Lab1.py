#Import libraries
import cv2
import numpy as np
from skimage import metrics as sm
import random

#SaltnPepper Noise
def saltAndPepper(anImage, prob):
    noisy_image = np.zeros(anImage.shape, np.uint8)
    for colIdx in range(anImage.shape[0]):
        for rowIdx in range(anImage.shape[1]):
            rand = random.random()
            if rand < prob:
                noisy_image[rowIdx][colIdx] = 0
            elif rand > (1 - prob):
                noisy_image[rowIdx][colIdx] = 255
            else:
                noisy_image[rowIdx][colIdx] = anImage[rowIdx][colIdx]


    return noisy_image

# Poisson Noise
def poisson_noise(image):
    noise = np.random.poisson(image).astype(np.uint8)
    noisy_image = image + noise
    return noisy_image

# Showing the image and Saving it
def display_image(name,image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.imwrite(f'{name}.jpg', image)
    cv2.waitKey()

if __name__ == '__main__':
    # read the image
    img = cv2.imread('ricknmorty.jpg', 1)
    # convert to grayscale image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # display the image
    display_image('Grayscale',img_gray)


    # Create Salt&Pepper Noise
    SnP_img = saltAndPepper(img_gray,0.10)
    # Show the image
    display_image('Image with Salt and Pepper' , SnP_img)

    # Create Poisson Noise
    Pois_img = poisson_noise(img_gray)
    # Show the image
    display_image('Image with Poisson Noise', Pois_img)

    # Kernel size
    ksize = (5, 5)

    # ---------------Filters Salt Pepper---------------
    img_gauss_blur = cv2.GaussianBlur(SnP_img, ksize, 0)
    display_image('Gaussian Blur with SnP', img_gauss_blur)
    img_median_blur = cv2.medianBlur(SnP_img, 5)
    display_image('Median Blur with SnP', img_median_blur)
    img_bilateral = cv2.bilateralFilter(SnP_img, 5, 20, 20)
    display_image('Bilateral Filter with SnP', img_bilateral)

    # ---------------Scores for Salt Pepper---------------
    # SSIM Score - Stractural Similarity Index Measure
    print("SSIM Score - Stractural Similarity Index Measure")
    score, _ = sm.structural_similarity(img_gray, img_gauss_blur, multichannel=True, full=True)
    print("Gaussian filter SSIM score: {:.4f}".format(score))
    score, _ = sm.structural_similarity(img_gray, img_median_blur, multichannel=True, full=True)
    print("Median filter SSIM score: {:.4f}".format(score))
    score, _ = sm.structural_similarity(img_gray, img_bilateral, multichannel=True, full=True)
    print("Bilateral filter SSIM score: {:.4f}\n".format(score))

    # MSE Score - Mean Squared Error
    print("MSE Score - Mean Squared Error")
    score = np.square(np.subtract(img_gray,img_gauss_blur)).mean()
    print("Gaussian filter MSE score: {:.4f}".format(score))
    score = np.square(np.subtract(img_gray, img_median_blur)).mean()
    print("Median filter MSE score: {:.4f}".format(score))
    score = np.square(np.subtract(img_gray, img_bilateral)).mean()
    print("Bilateral filter MSE score: {:.4f}\n".format(score))

    # ---------------Filters Poisson---------------
    img_gauss_blur = cv2.GaussianBlur(Pois_img, ksize, 0)
    display_image('Gaussian Blur with Poisson', img_gauss_blur)
    img_median_blur = cv2.medianBlur(Pois_img, 5)
    display_image('Median Blur with Poisson', img_median_blur)
    img_bilateral = cv2.bilateralFilter(Pois_img, 5, 20, 20)
    display_image('Bilateral Filter with Poisson', img_bilateral)

    # ---------------Scores for Poisson---------------
    # SSIM Score - Stractural Similarity Index Measure
    print("SSIM Score - Stractural Similarity Index Measure")
    score, _ = sm.structural_similarity(img_gray, img_gauss_blur, multichannel=True, full=True)
    print("Gaussian filter SSIM score: {:.4f}".format(score))
    score, _ = sm.structural_similarity(img_gray, img_median_blur, multichannel=True, full=True)
    print("Median filter SSIM score: {:.4f}".format(score))
    score, _ = sm.structural_similarity(img_gray, img_bilateral, multichannel=True, full=True)
    print("Bilateral filter SSIM score: {:.4f}\n".format(score))

    # MSE Score - Mean Squared Error
    print("MSE Score - Mean Squared Error")
    score = np.square(np.subtract(img_gray,img_gauss_blur)).mean()
    print("Gaussian filter MSE score: {:.4f}".format(score))
    score = np.square(np.subtract(img_gray, img_median_blur)).mean()
    print("Median filter MSE score: {:.4f}".format(score))
    score = np.square(np.subtract(img_gray, img_bilateral)).mean()
    print("Bilateral filter MSE score: {:.4f}\n".format(score))

    cv2.destroyAllWindows()
#Import libraries
import cv2
import numpy as np

def solarize(image,thresValue):
    # Check for threshold
    # if True return image else return complement(image)
    img_sabatier = np.where((image < thresValue), image, ~image)
    cv2.namedWindow(f'Solarized with Threshold {thresValue}', cv2.WINDOW_NORMAL)
    cv2.imshow(f'Solarized with Threshold {thresValue}', img_sabatier)
    cv2.imwrite(f'Image{thresValue}.jpg', img_sabatier)
    cv2.waitKey()

    
if __name__ == '__main__':
    # read the image
    img = cv2.imread('kenshin.jpg', 1)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.imshow('Original', img)
    cv2.waitKey()
    # convert to grayscale image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # display the image
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.imshow('Original', img_gray)
    cv2.imwrite('Gray Image.jpg', img_gray)

    #Apply Sabatier Effect - Solarization with different threshhold
    solarize(img_gray,64)
    solarize(img_gray, 128)
    solarize(img_gray, 192)

    #Clear All windows
    cv2.destroyAllWindows()


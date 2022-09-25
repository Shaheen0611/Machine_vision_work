import cv2
import numpy as np
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
# LOAD AN IMAGE USING 'IMREAD'
img = cv2.imread("../../Resources/lena.png")

height, width, channels = img.shape
emptyPicture = np.zeros((height, width, 3), dtype=np.uint8)


# Part 1.1 - Padding
def padding_image(_width):
    # 
    # TODO Code here
    #
    img_pad = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_REFLECT)
    return img_pad


# img_pad = padding_image(_width=100)


# Part 1.2 - Cropping
# x_0 = 80, y_0 = 80, x_1 = 382, y_1 = 382
def cropping_image(x_0, y_0, x_1, y_1):
    # 
    # TODO Code here
    #
    imgCropped = img[x_0:y_0, x_1:y_1]
    return imgCropped


#imgCropped = cropping_image(x_0=80, y_0=382, x_1=80, y_1=382)


# Part 1.3 - Resize
def resize_image(width, height):
    #
    # TODO Code here
    #
    imgResize = cv2.resize(img, (200, 200))

    return imgResize


#imgResize = resize_image(width=200, height=200)


# Part 2 - Copy
def copy_image():
    # 
    # TODO Code here
    #

    height, width, channels = img.shape
    imageCopy = np.zeros((height, width, 3), dtype=np.uint8)

    for x in range(height):
        for y in range(width):
            imageCopy[int(x)][int(y)] = img[int(x)][int(y)]

    return imageCopy


#imageCopy = copy_image()


# Part 3 - Gray 
def gray_image():
    # 
    # TODO Code here
    #
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray


#gray = gray_image()


# Part 4 - Shift color (with boundary at 0 and 255)
def shifting_colors(hue):
    # 
    # TODO Code here
    #

    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    emptyPicture = np.zeros((height, width, channels), dtype=np.uint8)


    for x in range(height):
        for y in range(width):
            emptyPicture[int(x)][int(y)] = img[int(x)][int(y)]

    return emptyPicture


emptyPicture = shifting_colors(hue=50)


# Part 5 - HSV
def hsf_image():
    # 
    # TODO Code here
    #

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    return hsv_img


hsv_img = hsf_image()


# Part 6 - Smoothing
def smoothing():
    # 
    # TODO Code here
    #
    smoothed_img = cv2.GaussianBlur(img,(15,15),cv2.BORDER_DEFAULT)

    return smoothed_img


#smoothed_img = smoothing()


# Part 7 - Rotation
def rotation():
    # 
    # TODO Code here
    #

    image90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    image180 = cv2.rotate(img, cv2.ROTATE_180)


    return image90, image180


image90, image180 = rotation()


def show_images():
    # cv2.imshow("Pdding image", img_pad)
    #cv2.imshow("Cropped image", imgCropped)
    #cv2.imshow("Resized image", imgResize)
    #cv2.imshow('Copied image', imageCopy)
    #cv2.imshow('Gray image', gray)
    cv2.imshow("Hue shifted image", emptyPicture)
    #cv2.imshow('HSV image', hsv_img)
    #cv2.imshow('Smoothed image', smoothed_img)
    #cv2.imshow('Rotated image 90', image90)
    #cv2.imshow('Rotated image 180', image180)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#show_images()


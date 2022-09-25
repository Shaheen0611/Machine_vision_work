import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

os.chdir(os.path.dirname(os.path.realpath(__file__)))
# LOAD AN IMAGE USING 'IMREAD'
img = cv2.imread("../../Resources/lena.png")
img2 = cv2.imread("../../Resources/lambo.png")
img3 = cv2.imread("../../Resources/cards.png")
img4 = cv2.imread("../../Resources/shapes.png")

pic1 = cv2.imread("../../Resources/Berkeley-test001-025/16077.jpg")
pic2 = cv2.imread("../../Resources/Berkeley-test001-025/24077.jpg")
pic3 = cv2.imread("../../Resources/Berkeley-test001-025/38092.jpg")
pic4 = cv2.imread("../../Resources/Berkeley-test001-025/42049.jpg")
pic5 = cv2.imread("../../Resources/Berkeley-test001-025/43074.jpg")
pic6 = cv2.imread("../../Resources/Berkeley-test001-025/45096.jpg")
pic7 = cv2.imread("../../Resources/Berkeley-test001-025/58060.jpg")
pic8 = cv2.imread("../../Resources/Berkeley-test001-025/76053.jpg")
pic9 = cv2.imread("../../Resources/Berkeley-test001-025/78004.jpg")
pic10 = cv2.imread("../../Resources/Berkeley-test001-025/86000.jpg")
pic11 = cv2.imread("../../Resources/Berkeley-test001-025/89072.jpg")
pic12 = cv2.imread("../../Resources/Berkeley-test001-025/101085.jpg")
pic13 = cv2.imread("../../Resources/Berkeley-test001-025/119082.jpg")
pic14 = cv2.imread("../../Resources/Berkeley-test001-025/126007.jpg")
pic15 = cv2.imread("../../Resources/Berkeley-test001-025/156065.jpg")
pic16 = cv2.imread("../../Resources/Berkeley-test001-025/157055.jpg")
pic17 = cv2.imread("../../Resources/Berkeley-test001-025/163085.jpg")
pic18 = cv2.imread("../../Resources/Berkeley-test001-025/167062.jpg")
pic19 = cv2.imread("../../Resources/Berkeley-test001-025/170057.jpg")
pic20 = cv2.imread("../../Resources/Berkeley-test001-025/175032.jpg")
pic21 = cv2.imread("../../Resources/Berkeley-test001-025/219090.jpg")
pic22 = cv2.imread("../../Resources/Berkeley-test001-025/220075.jpg")
pic23 = cv2.imread("../../Resources/Berkeley-test001-025/295087.jpg")
pic24 = cv2.imread("../../Resources/Berkeley-test001-025/296007.jpg")
pic25 = cv2.imread("../../Resources/Berkeley-test001-025/300091.jpg")

height, width, channels = img.shape
emptyPicture = np.zeros((height, width, 3), dtype=np.uint8)

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:5] == imgArray[0][0].shape[:5]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:5] == imgArray[0].shape[:5]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 5: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


# Example to use stackfunction
imgStack = stackImages(0.5, ([pic1, pic2, pic3, pic4, pic5], [pic6, pic7, pic8, pic9, pic10],
                             [pic11, pic12, pic13, pic14, pic15],[pic16, pic17, pic18, pic19, pic20],
                             [pic21, pic22, pic23, pic24, pic25],))


def sobel_edge_detection(imgStack):
    img_blur = cv2.GaussianBlur(imgStack, (3, 3), 0)
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)  # Combined X and Y Sobel Edge Detection

    sobel_edge=sobelxy

    return sobel_edge
sobel_edge = sobel_edge_detection(imgStack)

def canny_edge_detection(imgStack):
    img_gray = cv2.cvtColor(imgStack, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    canny_edge = cv2.Canny(image=img_blur, threshold1=50, threshold2=50)

    return canny_edge
canny_edge = canny_edge_detection(imgStack)


def downsizing():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    img2 = cv2.imread("../../Resources/lambo.png")
    cv2.waitKey(0)
    rows, cols, _channels = map(int, img2.shape)
    img2down = cv2.pyrDown(img2, dstsize=(2 // cols, 2 // rows))

    return img2down

img2down = downsizing()


def upsizing():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    img2 = cv2.imread("../../Resources/lambo.png")



    cv2.waitKey(0)
    rows, cols, _channels = map(int, img2.shape)
    img2up = cv2.pyrUp(img2, dstsize=(2 * cols, 2 * rows))

    return img2up

img2up = upsizing()


def template_match():
    img = cv2.imread("../../Resources/Berkeley-test001-025/167062.jpg")
    img_temp = cv2.imread("../../Resources/Berkeley-test001-025/167062_template.png")
    w, h = img_temp.shape[:-1]
    res = cv2.matchTemplate(img, img_temp, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + h, top_left[1] + w)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
    return img


x = template_match()

cv2.imshow('img stack', imgStack)
cv2.imshow('Sobel edge detection', sobel_edge)
cv2.imshow('Canny edge detection', canny_edge)
cv2.imshow('Downsized lambo', img2down)
cv2.imshow('Upsized lambo', img2up)
cv2.imshow('Template matching', x)

cv2.waitKey(0)
cv2.destroyAllWindows()


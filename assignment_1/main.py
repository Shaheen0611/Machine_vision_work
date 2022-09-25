import cv2

def main():
    img = cv2.imread('lena.png', cv2.IMREAD_COLOR)
    cv2.imshow('image', img)


    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    pixel = height * width
    imgdata = img.dtype

    print(f'Image height:', height)
    print(f'Image width:', width)
    print(f'Number of channels: ', channels)
    print(f'Image size:', pixel)
    print(f'Image data type is:', imgdata)

    vid = cv2.VideoCapture(0)
    fps = vid.get(cv2.CAP_PROP_FPS)
    vidwidth = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    vidheight = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f'Video FPS:', fps)
    print(f'Video Height:', vidheight)
    print(f'Video Width:', vidwidth)

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

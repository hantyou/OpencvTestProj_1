import cv2 as cv

print('cv\'s version is ' + cv.__version__)
cv.namedWindow('Image_I', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
Im = cv.imread('lena512color.tiff')
cv.imshow('Image_I', Im)
cv.waitKey(0)

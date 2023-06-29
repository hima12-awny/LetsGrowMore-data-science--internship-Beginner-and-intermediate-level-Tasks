import cv2 as cv
print('cv2 optimized py gpu:', cv.useOptimized())

img            = cv.imread('pic.jpg')

imggry         = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

inverted_img   = cv.bitwise_not(imggry)

imgblur        = cv.GaussianBlur(inverted_img, (29, 29), 0)

inverted_img_b = cv.bitwise_not(imgblur)

sketch         = cv.divide(imggry, inverted_img_b, scale=256.0)

cv.imshow('results',        sketch)
cv.imshow('inverted_img_b', inverted_img_b)
cv.imshow('blured img',     imgblur)
cv.imshow('inverted img',   inverted_img)
cv.imshow('gray scale img', imggry)
cv.imshow('normal img',     img)

cv.waitKey(0)
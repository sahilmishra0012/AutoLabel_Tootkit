from cv2 import cv2

image = cv2.imread('/home/samkiller007/Desktop/test.jpeg')

height = image.shape[0]
width = image.shape[1]

if height>width:
    reduction_factor = (float)(height/224.0)
    height = 224
    width = (float)(width/reduction_factor)
    height = round(height)
    width = round(width)
    img = cv2.resize(image, (width,height))
    if (224-width)%2!=0:
        left = int((224-width)/2)
        right = left+1
    else:
        left = (224-width)/2
        right = left
    print("Width",(224-width)/2)
    img = cv2.copyMakeBorder( img, 0, 0, left, right, cv2.BORDER_CONSTANT)

elif height<width:
    reduction_factor = (float)(width/224.0)
    width = 224
    height = (float)(height/reduction_factor)
    height = round(height)
    width = round(width)
    print("Height",height)
    img = cv2.resize(image, (width,height))
    if (224-height)%2!=0:
        top = int((224-height)/2)
        bottom = left+1
    else:
        top = (224-height)/2
        bottom = left
    print("Width",(224-width)/2)
    img = cv2.copyMakeBorder( img, top, bottom, 0, 0, cv2.BORDER_CONSTANT)

elif height==width:
    height = 224
    width = 224
    img = cv2.resize(image, (width,height))

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite('image9211.jpeg',img)

print(img.shape)
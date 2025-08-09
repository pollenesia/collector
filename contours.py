import cv2

img = cv2.imread('5/image304.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.bitwise_not(gray)

gray1 = cv2.GaussianBlur(gray, (235, 235), 0)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)
# gray -= gray1 // 2
gray = cv2.bitwise_not(gray)

_, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

# thresh = cv2.Canny(gray, 10, 80)

# thresh = cv2.adaptiveThreshold(
# gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 37, 1)

contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for cnt in contours:
    area = cv2.contourArea(cnt)
    if area >= 4 and area < 500:  # Filter small noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
        # Or: cv2.drawContours(img, [cnt], -1, (255,0,0), 2)

cv2.imshow('Prepared', gray)
cv2.imshow('Particles Distinguished', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

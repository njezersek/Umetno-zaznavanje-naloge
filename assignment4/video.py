import cv2

video = cv2.VideoCapture("data/video.mp4")


fast = cv2.FastFeatureDetector_create()
fast.setNonmaxSuppression(0)
fast.setThreshold(10)

while video.isOpened():
    ret, frame = video.read()
    if ret == True:
        key_points = fast.detect(frame, None)
        cv2.imshow('video', cv2.drawKeypoints(frame, key_points, frame))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
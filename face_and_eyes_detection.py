import cv2 as cv

def detectAndDisplayHumanFaceAndEyes(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray, 1.1, 4)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

    cv.imshow('Capture - Face and Eyes detection', frame)

def detectAndDisplayCatFace(frame):
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    #-- Detect faces
    faces = cat_face_cascade.detectMultiScale(frame_rgb, 1.1, 4)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

    cv.imshow('Capture - Face detection', frame)

face_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eyes_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')

cat_face_cascade = cv.CascadeClassifier('data/haarcascades/haarcascade_frontalcatface.xml')
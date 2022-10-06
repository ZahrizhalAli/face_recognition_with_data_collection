import cv2
import os

video = cv2.VideoCapture(1)

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0

nameId = str(input('Enter your Name:')).lower()

path = 'images/' + nameId

isExist = os.path.exists(path)

if isExist:
    print(f"{nameId} Already Exist")
    nameId = str(input('Enter your name again: '))
else:
    os.makedirs(path)

while True:
    ret, frame = video.read()
    faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for x,y,w,h in faces:
        count = count + 1
        name=f'./images/{nameId}/{count}.jpg'
        print(f"Creating images...{name}")
        cv2.imwrite(name, frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    cv2.imshow('Window Frame', frame)
    key = cv2.waitKey(1)
    if count > 200:
        break
video.release()
cv2.destroyAllWindows()
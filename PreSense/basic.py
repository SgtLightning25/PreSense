import cv2
import numpy as np
import face_recognition

imgZhao = face_recognition.load_image_file('ImagesBasic/Dave Zachary Macarayo.jpg')
imgZhao = cv2.cvtColor(imgZhao, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Anferny Glenn Gulle.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgZhao)[0]
encodeZhao = face_recognition.face_encodings(imgZhao)[0]
cv2.rectangle(imgZhao, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceTest[3], faceTest[0]), (faceTest[1], faceTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeZhao], encodeTest)
faceDis = face_recognition.face_distance([encodeZhao], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Zhao Jinmai', imgZhao)
cv2.imshow('Wrench Ivan Borja', imgTest)
cv2.waitKey(0)

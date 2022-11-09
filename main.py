import cv2
import face_recognition as fr
import os
import numpy as np


path = "image"
className = []
images = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])

def Mahoa(images):
    encodeList = []
    for img in images:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListknow = Mahoa(images)
print("ma hoa thanh cong " ,len(encodeListknow))

cap = cv2.VideoCapture(0)

while True :
    ret,frame = cap.read()
    framS = cv2.resize(frame,(0,0),None,fx=0.5,fy=0.5)
    framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)
    #xác định face
    facecurframe= fr.face_locations(framS)
    encodecurFrame = fr.face_encodings(framS)
    for encodeface,faceLoc in zip(encodecurFrame,facecurframe):

        maches = fr.compare_faces(encodeListknow,encodeface)
        faceDis = fr.face_distance(encodeListknow,encodeface)
        print(faceDis)
        machIndex = np.argmin(faceDis) # đẩy về vị trí index nhỏ nhất

        if faceDis[machIndex] < 0.50:
            name = className[machIndex].upper()
        else:
            name = "khong biet"

        y1,x2,y2,x1 = faceLoc
        y1,x2,y2,x1 = y1*2 ,x2*2 ,y2*2 ,x1*2
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(frame,name,(x2,y2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,215,0),1)

    cv2.imshow("face",frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
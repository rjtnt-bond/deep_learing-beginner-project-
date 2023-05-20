import face_recognition
import cv2
import numpy as np
import csv
import os
import glob
from datetime import datetime

video_capture=cv2.VideoCapture(0)
bill_image=face_recognition.load_image_file("photos/bil.jpg")
bill_encoding=face_recognition.face_encodings(bill_image)[0]

eines_image=face_recognition.load_image_file("photos/eines.jpg")
eines_encoding=face_recognition.face_encodings(eines_image)[0]

mark_image=face_recognition.load_image_file("photos/mark.jpg")
mark_encoding=face_recognition.face_encodings(mark_image)[0]


Newton_image=face_recognition.load_image_file("photos/Newton.jpg")
Newton_encoding=face_recognition.face_encodings(Newton_image)[0]

know_face_encoding=[
    
 bill_encoding,
 eines_encoding,
 mark_encoding,
 Newton_encoding
]

known_faces_names=[

    "Bill gates",
    "Albert Einstein",
    "Mark",
    "Newton"
]

students_list=known_faces_names.copy()

face_locatios=[]
face_encodings=[]
face_names=[]
s=True

now=datetime.now()

current_date=now.strftime("%Y-%m-%d")

f=open(current_date+'.csv','w+',newline="")
lnwrite=csv.writer(f)

while True:
    _,frame=video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_face=small_frame[:,:,::-1]
    if s:
        face_locatios=face_recognition.face_locations(rgb_small_face)
        face_encodings=face_recognition.face_encodings(rgb_small_face,face_locatios)
        face_names=[]

        for face_encoding in face_encodings:
            matches=face_recognition.compare_faces(know_face_encoding,face_encoding)
            name=""
            face_distance=face_recognition.face_distance(know_face_encoding,face_encoding)
            best_match_index=np.argmin(face_distance)
            if matches[best_match_index]:
                name=known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name in students_list:
                    students_list.remove(name)
                    print(name)
                    current_time=now.strftime("%H-%M-%S")
                    lnwrite.writerow([name,current_time])
                    
    cv2.imshow("class attendence",frame)
    if cv2.waitKey(1)& 0xFF==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
f.close()

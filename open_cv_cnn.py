import cv2
import face_recognition
import numpy as np

cap = cv2.VideoCapture(0)

fluke_image = face_recognition.load_image_file("infomation/1.jpg")
karn_image = face_recognition.load_image_file("infomation/karn.jpg")
AjRach_image = face_recognition.load_image_file("infomation/AjRach.jpg")
ajMin_image = face_recognition.load_image_file("infomation/ajMin.jpg")

fluke_face_encoding = face_recognition.face_encodings(fluke_image)[0]
karn_face_encoding = face_recognition.face_encodings(karn_image)[0]
AjRach_face_encoding = face_recognition.face_encodings(AjRach_image)[0]
ajMin_face_encoding = face_recognition.face_encodings(ajMin_image)[0]

known_face_encodings = [
    ajMin_face_encoding,
    AjRach_face_encoding,
    karn_face_encoding,
    fluke_face_encoding
]
known_face_names = [
    "Aj. Min",
    "Aj. Rachasak",
    "Karn",
    "Fluke"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
 
while True:
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 2.0)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            print(face_distances)
            best_match_index = np.argmin(face_distances)
            print(best_match_index)
            print(matches)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

    # t = 0
    # r = 0
    # b = 0
    # l = 0
    # print(rgb_frame)
    # face_locations = face_recognition.face_locations(rgb_frame)
    # # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # # print(gray)
    # for top, right, bottom, left in face_locations:
    #     # print("top: "+str(top))
    #     # print("right: "+str(right))
    #     # print("bottom: "+str(bottom))
    #     # print("left: "+str(left))
    #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    #     cv2.putText(frame,"Fluke",(left,top-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1)
    #     # id,_ = clf.predict(gray[t:b,l:r])
    #     # if id == 1:
    #         # cv2.putText(frame,"Fluke",(left,top-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1)
    #     t = top
    #     r = right
    #     b = bottom
    #     l = left

    
    # if len(face_locations) == 1:
    #     img_id = img_id+1
    #     # result = frame[l:l+t, r:r+b]
    #     result = frame[t:b, l:r]
    #     create_dataset(result,id,img_id)
        

    # frame = cv2.resize(frame, (1200, 720))     
    

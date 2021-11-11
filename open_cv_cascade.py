import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
# detector = MTCNN()
def check_confidence(id, confidence):
    print(str(id))
    text = "Unknow"
    if id == 1:
        if confidence <= 115:
            text = 'Fluke'
    elif id == 2:
        if confidence <= 115:
            text = 'Kan Atthakorn'
    elif id == 3:
        if confidence <= 115:
            text = 'AJ Rach'
    elif id == 4:
        if confidence <= 115:
            text = 'Pakkard'
    elif id == 5:
        if confidence <= 115:
            text = 'Frame'
    elif id == 6:
        if confidence <= 115:
            text = 'Arm'

    print(str(" {0}%".format(round(100-confidence))))
    return text

def create_dataset(img, id, img_id):
    cv2.imwrite("data/pic."+str(id)+"."+str(img_id)+".jpg",img)

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray,scaleFactor,minNeighbors)
    coords = []
    for (x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        id,confidence = clf.predict(gray[y:y+h, x:x+w])
        text = check_confidence(id, confidence)
        cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
        coords = [x,y,w,h]
    return img,coords

def detect(img, faceCascade, clf):
    img,coords = draw_boundary(img, faceCascade, 1.1, 10, (0,0,255), clf)
    if len(coords) == 4:
        img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]

    return img


cap = cv2.VideoCapture(0)
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read('classifier.xml')

while (True):
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    frame = detect(small_frame, faceCascade, clf)
    cv2.imshow('frame', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()

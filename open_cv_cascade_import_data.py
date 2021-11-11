import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
# detector = MTCNN()

def create_dataset(img, id, img_id):
    cv2.imwrite("data/pic."+str(id)+"."+str(img_id)+".jpg",img)

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # detector.detect_faces(gray)
    features = classifier.detectMultiScale(gray,scaleFactor,minNeighbors)
    coords = []
    for (x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1)
        coords = [x,y,w,h]
    return img,coords

def detect(img, faceCascade):
    img,coords = draw_boundary(img, faceCascade, 1.1, 10, (0,0,255), "Face")
    if len(coords) == 4:
        # print(coords)
        result = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        create_dataset(result,id,img_id)

    return img


img_id = 0
id=2
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('kan_anthakorn.mov')

while (True):
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # rgb_small_frame = small_frame[:, :, ::-1]
    img_id+=1
    frame = detect(small_frame, faceCascade)
    cv2.imshow('frame', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()

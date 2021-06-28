import cv2

face_cascasde = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#started the video capture
video = cv2.VideoCapture(0)

while True:
    #read the video object
    check, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascasde.detectMultiScale(gray,
    scaleFactor = 1.1,
    minNeighbors = 5
    )

    #making rectqangle around he faces co-ordiantes
    for x, y, w, h in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)

    #display output in window 
    cv2.imshow("Detecting", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

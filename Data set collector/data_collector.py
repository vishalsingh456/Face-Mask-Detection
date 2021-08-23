import cv2
import numpy
import os


har_cascad = cv2.CascadeClassifier('./requirement/haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier("./requirement/haarcascade_eye.xml")

cam = cv2.VideoCapture(0)
count = 1

name = input("Enter your name: ")
try:
    os.mkdir(name)

except:
    pass

typ = input("Enter the type: ")
try:
    os.mkdir(f"./{name}/{typ}")
except:
    pass


while True:
    flag,img = cam.read()
    
    #check weather the image is read properly or not
    if flag:
        
        #converting the image from BGR to gray color
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        faces = har_cascad.detectMultiScale(gray, 1.1, 5)
        
        
        #read x_axis, y_axis, width and height of faces and draw a rextangle around it
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        #chech weather the face is recognised or not and also weather it has detected a single face or multiple
        if faces != () or len(faces) ==1:
           
            #crop the image as per the locion provided by the faces
            img2 = img[y:y+h ,x:x+w]
            #gray scal image 
            gray = img[y:y+h ,x:x+w]
            flag = eye.detectMultiScale(gray)
            if flag != ():
                count+=1
                #saving the cropped face
                cv2.imwrite(f"./{name}/{typ}/{name}{count}.jpg",img2)
            
    #writing the image count on the output screen
    cv2.putText(img, str(count), (150, 150), cv2.FONT_HERSHEY_PLAIN,1, (0, 0, 224),2)
    
    #showing the image
    cv2.imshow('img', img)
    if cv2.waitKey(1)==27 or count>=400:
        break
        

cam.release()
cv2.destroyAllWindows()
input("press enter to continue")
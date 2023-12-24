from keras.models import load_model
import numpy as np
import cv2
cap=cv2.VideoCapture(0)

model=load_model('sign_mnist_cnv_50_Epochs.h5')

def getLetter(result):
    classLabels = { 0: 'A',
                    1: 'B',
                    2: 'c',
                    3: 'D',
                    4: 'E',
                    5: 'F',
                    6: 'G',
                    7: 'H',
                    8: 'I',
                    9: 'J',
                    10: 'K',
                    11: 'L',
                    12: 'M',
                    13: 'N',
                    14: 'O',
                    15: 'P',
                    16: 'Q',
                    17: 'R',
                    18: 'S',
                    19: 'T',
                    20: 'U',
                    21: 'V',
                    22: 'W',
                    23: 'X',
                    24: 'Y',
                    25: 'Z'}
    try:
        res=int(result)
        print(classLabels[res],end="")
        return classLabels[res]
    except:
        return "Error"

while True:
    ret, frame= cap.read()
    
    roi=frame[100:400, 320:620]
    cv2.imshow('roi',roi)
    roi=cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi=cv2.resize(roi, (28,28), interpolation= cv2.INTER_AREA)
    
    
    cv2.imshow('roi scaled and gray', roi)
    copy=frame.copy()
    cv2.rectangle(copy,(320, 100), (620, 400), (255,0,0),5)
    
    roi=roi.reshape(1,28,28,1)

    result=str(np.argmax(model.predict(roi, 1, verbose=0)[0]))
    #print(result,end="")
 

    cv2.putText(copy, getLetter(result),(300, 100), cv2.FONT_HERSHEY_COMPLEX, 2,(0, 255,0),2)
    cv2.imshow('frame',copy)
    
    if cv2.waitKey(2) == 13:
        break
cap.release()
cv2.destroyAllWindows()


'''

   result=str(model.predict_classes(roi, 1, verbose=0)[0])

   '''
import cv2 as cv
from keras.models import load_model
import numpy as np
import glob as gb
paths=gb.glob('C:/Enlighten here/Essentials/Machine Learning/MyProjects/Kaggle-Paper Rock Scissors/Data/rock/*jpg')
model=load_model('C:/Enlighten here/Essentials/Machine Learning/MyProjects/Kaggle-Paper Rock Scissors/best_model.hdf5')
cap=cv.VideoCapture(0)
(R,G,B)=(205.285,205.295,200.487)
out = cv.VideoWriter('C:/Enlighten here/Essentials/Machine Learning/MyProjects/Kaggle-Paper Rock Scissors/Sample.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
player1=input('Enter the name of the first player:')
player2=input('Enter the name of the second player:')

def getShape(pred):
    [p,s,sc,emp]=pred[0]
    mx=[p,s,sc,emp].index(max([p,s,sc,emp]))
    if len(set([p,s,sc,emp]))!=2:
        return 'Empty'
    if mx==0:
        return 'Paper'
    elif mx==1:
        return 'Stone'
    elif mx==2:
        return 'Scissor'

def game(class1,class2):
    global player1,player2
    if class1=='Empty' or class2=='Empty':
        return 'Wait'
    elif class2==class1:
        return 'Draw'
    elif class1=='Paper' and class2=='Stone':
        return player1
    elif class1=='Scissor' and class2=='Paper':
        return player1
    elif class1=='Stone' and class2=='Scissor':
        return player1
    else:
        return player2
while True:
    ret,frame=cap.read()
    roi = frame[80:280, 30:230]
    cv.rectangle(frame, (30, 80), (230, 280), (255, 0, 0))
    frame = cv.flip(frame, 1)
    poi=frame[80:280,30:230]
    poi=cv.flip(poi,1)
    cv.rectangle(frame, (30, 80), (230, 280), (0, 255, 0))
    roi=cv.cvtColor(roi,cv.COLOR_BGR2RGB)
    roi=cv.resize(roi,(128,128))
    roi=np.reshape(roi,(1,128,128,3))
    poi = cv.cvtColor(poi, cv.COLOR_BGR2RGB)
    poi = cv.resize(poi, (128, 128))
    poi = np.reshape(poi, (1, 128, 128, 3))
    pred2=model.predict(roi)
    pred1=model.predict(poi)
    class1=getShape(pred1)
    class2=getShape(pred2)
    frame[400:,]=np.ones((80,640,3))
    cv.putText(frame, player1, (30, 450), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame, player2, (530, 450), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    winner = game(class1, class2)
    if winner!='Wait':
        cv.putText(frame, class1, (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(frame, class2, (530, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if winner=='Draw':
            cv.putText(frame, 'DRAW', (260, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv.putText(frame,winner, (250, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.putText(frame, 'WINS', (255, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.imshow('frame', frame)
    out.write(frame)
    if cv.waitKey(1)==27:
        img=np.ones((480,640,3))/255.0
        cv.putText(img, 'Thank You', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv.putText(img, ':-)', (50, 150), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv.imshow('frame',img)
        cv.waitKey(2000)
        break
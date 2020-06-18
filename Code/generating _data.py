import cv2 as cv
PATH=''# Enter the local folder path where data will be generated
gestures=list(('empty','rock','paper','scissors'))
count=0
cap=cv.VideoCapture(0)
def showtime(folder):
    count=0
    while count<170:
        ret,shot=cap.read()
        roi=shot[100:300,100:300]
        roi = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
        cv.rectangle(shot, (100, 100), (300, 300), (255, 0, 0))
        shot[100:300, 100:300] = roi
        shot = cv.flip(shot, 1)
        cv.putText(shot,f'Make the sign for {folder}',(10,80),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
        cv.imshow('frame', shot)
        if count==50:
            cv.putText(shot, 'Starts in 1..', (100, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if count==100:
            cv.putText(shot, 'Starts in 1..2..', (100, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if count==150:
            cv.putText(shot, 'Starts in 1..2..3..', (100, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if count==160:
            cv.putText(shot, 'Starts in 1..2..3..GO', (100, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 2)
        cv.imshow('frame', shot)
        cv.waitKey(1)
        count+=1
        print(f'{count} wait.....')
for folder in gestures:
    showtime(folder)
    count=1
    while True:
        ret,frame=cap.read()
        if not ret:
            print('[INFO] ERROR')
            break
        roi=frame[100:300,100:300]
        roi=cv.cvtColor(roi,cv.COLOR_BGR2RGB)
        cv.rectangle(frame,(100,100),(300,300),(255,0,0))
        frame[100:300,100:300]=roi
        frame = cv.flip(frame, 1)
        cv.imshow('frame',frame)
        roi=cv.resize(roi,(128,128))
        cv.imwrite(f'PATH/{folder}/{folder}{count}.jpg', roi)# Here word PATH is not part of the path, but the representation of the above initialised PATH
        cv.waitKey(10)
        count+=1
        print(f'[INFO] {count} images captured...')
        if count==1500:
            break

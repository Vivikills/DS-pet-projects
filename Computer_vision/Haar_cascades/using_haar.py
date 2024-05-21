import cv2
import numpy as np

#Для распознавания людей, которые идут от камеры (у них видно все тело, но не видно лица)
cascade_path_fullperson = 'haarcascade_fullbody.xml'

#Для распознавания людей, которые стоят (или идут) лицом к камере
cascade_path_frontface = 'haarcascade_frontalface_alt.xml'

#Для людей, которые повернуты к камере профильной стороной лица
cascade_path_profileface = 'haarcascade_profileface.xml'

#Т.к. на камере плохо видны лица людей, вдобавок подключаем каскад для верхней части тела
cascade_path_upperbody = 'haarcascade_upperbody.xml'

#Используем класс для детекции объектов, основанный на каскадах
clf1 = cv2.CascadeClassifier(cascade_path_fullperson)
clf2 = cv2.CascadeClassifier(cascade_path_frontface)   
clf3 = cv2.CascadeClassifier(cascade_path_profileface)
clf4 = cv2.CascadeClassifier(cascade_path_upperbody)

#Захват видео
video = cv2.VideoCapture('crowd.mp4')

if video.isOpened():
    fps = video.get(5)
    print('Фреймов в секунду: ', fps,'FPS')
    frame_count = video.get(7)
    print('Частота кадров: ', frame_count)
    frame_width = int(video.get(3))
    print('Ширина кадров: ', frame_width)
    frame_height = int(video.get(4))
    print('Высота кадров: ', frame_height)
else:
    print('Ошибка открытия видео')

#Для записи видео с отрисованными людьми
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('detection_crowd.mp4', fourcc, fps, (frame_width, frame_height))

while video.isOpened():
    ret, frame = video.read()

    if ret == True:

        #Переводим фрейм в черно-белый, так лучше работают каскады
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Детектирование
        faces_front = clf2.detectMultiScale(
            gray,
            scaleFactor = 1.03,
            minNeighbors = 3,
            maxSize= (30,30)   
        )

        fullpersons = clf1.detectMultiScale(
            gray,
            scaleFactor = 1.03,
            minNeighbors = 4,
            minSize= (60,100),
            maxSize= (200,400)   
        )

        faces_profile = clf3.detectMultiScale(
            gray,
            scaleFactor = 1.03,
            minNeighbors = 4,
            maxSize= (30,30)   
        )

        upperpersons = clf4.detectMultiScale(
            gray,
            scaleFactor = 1.03,
            minNeighbors = 5,
            minSize=(50,50),
            maxSize= (200,200)   
        )

        #Отрисовка прямоугольников
        for (x,y,h,w) in fullpersons:
            cv2.rectangle(frame, (x,y), (x+h,y+w), (255,255,0), 2)

        for (x1,y1,h1,w1) in faces_profile:
            cv2.rectangle(frame, (x1,y1), (x1+h1,y1+w1), (255,255,0), 2)
    
        for (x2,y2,h2,w2) in faces_front:
            cv2.rectangle(frame, (x2,y2), (x2+h2,y2+w2), (255,255,0), 2)

        for (x3,y3,h3,w3) in upperpersons:
            cv2.rectangle(frame, (x3,y3), (x3+h3,y3+w3), (255,255,0), 2)
    else:
        print('Ошибка чтения видео')

    #Показываем видео с отрисованными людьми
    cv2.imshow('Crowd', frame) 

    #Записываем видео с отрисованными людьми
    out.write(frame)

    key = cv2.waitKey(1)

    #Клавиша остановки программы
    if key == ord('q'):
      break
        
        
video.release()
out.release
cv2.destroyAllWindows()

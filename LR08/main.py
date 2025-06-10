import cv2
import numpy as np

def higdhlightFace(net, frame, conf_threshold=0.7):
    ''' Функция определения лиц'''
    # Копия текущего кадра
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth  = frameOpencvDnn.shape[1]

    # Преобразование картинки в двоичный пиксельный объект
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300,300))

    # Установка входного параметра нейросети
    net.setInput(blob)
    detection = net.forward()

    # Идентификация
    faceBoxes = []
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence > conf_threshold:
            # Определение координат рамки
            x1 = int(detection[0,0,i,3]*frameWidth)
            y1 = int(detection[0,0,i,4]*frameHeight)
            x2 = int(detection[0,0,i,5]*frameWidth)
            y2 = int(detection[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])

            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), 
                          int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

def video_detector():
# Получение видео-информации
    video = cv2.VideoCapture(0)
    while cv2.waitKey(1)<0:
        hasFrame, frame = video.read()

        if not hasFrame:
            cv2.waitKey()
            break
        resultImg, faceBoxes = higdhlightFace(faceNet, frame)

        if not faceBoxes:
            print('Лица не распознаны')

        cv2.imshow("Face detection", frame)
    cv2.destroyAllWindows()

def image_detector(path):
    #Получение информации из изображения
    image = cv2.imread(path)
    frame = np.array(image)

    resultImg, faceBoxes = higdhlightFace(faceNet, frame)

    if not faceBoxes:
        print('Лица не распознаны')

    cv2.imshow("Face detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Загрузка нейросети и весов для распознавания лиц
faceProto = 'LR08\opencv_face_detector.pbtxt'
faceModel = 'LR08\opencv_face_detector_uint8.pb'
faceNet = cv2.dnn.readNet(faceModel, faceProto)

path = 'LR08/face.jpg'
#path = 'LR08/group.jpg'

image_detector(path)

#video_detector()
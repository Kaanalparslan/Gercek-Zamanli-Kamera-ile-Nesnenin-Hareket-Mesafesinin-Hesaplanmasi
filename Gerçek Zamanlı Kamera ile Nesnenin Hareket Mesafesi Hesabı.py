"""
Bu proje yolov5s kütüphanesinin algıladığı nesnelerin merkezini hesaplayıp ekrandaki noktaya olan mesafesini hesaplar.
Çalıştırdığınız zaman telefon, su şişesi, mouse gibi nesneleri gösterek deneyebilirsiniz. q'ya basarak programı kapatabilirsiniz.

Kodun çalışması için OPENCV, NUMPY, PYTORCH VE ULTRALYTICS kütüphanelerinin olması gereklidir. 
Eğer çalışmazsa ultralytics sitesine giderek yolov5s dokümantasyonunu indirmeniz gereklidir.

-----------------------------------------------------------------------------------------------------------------------

YOLOV5'in algıladığı nesneler:

person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat
traffic light, fire hydrant, stop sign, parking meter, bench
cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, 
kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket
bottle, wine glass, cup, fork, knife, spoon, bowl
banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
chair, sofa, pottedplant, bed, diningtable, toilet, tvmonitor, laptop, mouse, 
remote, keyboard, cell phone, microwave, oven, toaster, sink, 
refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

-----------------------------------------------------------------------------------------------------------------------

"""

import cv2  # OpenCV kütüphanesini içe aktarır
import numpy as np  # NumPy kütüphanesini içe aktarır
import torch  # PyTorch kütüphanesini içe aktarır

# YOLOv5 modelini PyTorch Hub'dan yükler
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Kamerayı başlatır
cap = cv2.VideoCapture(0)

# İlk kareyi okur ve yatay olarak çevirir
_, prev = cap.read()
prev = cv2.flip(prev, 1)

# Yeni kareyi okur ve yatay olarak çevirir
_, new = cap.read()
new = cv2.flip(new, 1)

# Referans noktayı tanımlar (ekranın sağ üst köşesinde bir nokta)
ref_point = (prev.shape[1] - 10, 10)

# Ekran ve kamera çözünürlüklerini tanımlar
screen_resolution = (1920, 1080)
camera_resolution = (1280, 720)

# Pikselden santimetreye dönüştürme oranını hesaplar
pixel_to_cm = 0.026458333333333 * (camera_resolution[0] / screen_resolution[0])

# Nesnenin merkezine göre mesafeyi pisagor teoremi ile hesaplayan fonksiyon
def calculate_distance(object_center):
    distance_px = ((object_center[0] - ref_point[0]) ** 2 + (object_center[1] - ref_point[1]) ** 2) ** 0.5
    distance_cm = distance_px * pixel_to_cm
    return distance_cm

# Sonsuz döngü
while True:
    diff = cv2.absdiff(prev, new)                                                 # İki kare arasındaki farkı hesaplar
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)                                 # Farkı gri tonlamaya çevirir
    diff = cv2.blur(diff, (5, 5))                                                 # Bulanıklaştırma uygular
    _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)                   # İkili eşikleme uygular
    thresh = cv2.dilate(thresh, None, 3)                                          # Dilatasyon uygular
    thresh = cv2.erode(thresh, np.ones((4, 4)), 1)                                # Erozyon uygular
    contor, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Konturları bulur
    
    # Önceki karedeki nesneleri YOLOv5 ile algılar
    results = model(prev)
    boxes = results.pred[0][:, :4]  # Algılanan nesnelerin sınırlayıcı kutularını alır

    # Algılanan her nesne için
    for box in boxes:
        x_min, y_min, x_max, y_max = box.cpu().numpy().astype(int)  # Sınırlayıcı kutunun koordinatlarını alır
        
        # Koordinatları geçerli aralıkta tutar
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, prev.shape[1])
        y_max = min(y_max, prev.shape[0])

        # Sınırlayıcı kutuyu çizer
        cv2.rectangle(prev, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Nesnenin merkez noktasını hesaplar
        object_center_x = (x_min + x_max) // 2
        object_center_y = (y_min + y_max) // 2

        # Referans noktayı ve nesne merkezini çizer
        cv2.circle(prev, ref_point, 5, (0, 0, 255), -1)
        cv2.circle(prev, (object_center_x, object_center_y), 5, (0, 0, 255), -1)

        # Mesafeyi hesaplar ve ekrana yazdırır
        distance_cm = calculate_distance((object_center_x, object_center_y))
        cv2.putText(prev, "{:.2f} cm".format(distance_cm), (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Sonucu gösterir
    cv2.imshow("Kamerada Nesne Mesafesi Hesabı", prev)

    # Yeni kareyi önceki kare yapar ve yeni kareyi okur
    prev = new
    _, new = cap.read()
    new = cv2.flip(new, 1)

    # 'q' tuşuna basıldığında döngüden çıkar
    if cv2.waitKey(1) == ord('q'):
        break

# Kamerayı serbest bırakır ve pencereleri kapatır
cap.release()
cv2.destroyAllWindows()
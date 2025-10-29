# Import OpenCV untuk image processing
import cv2
import numpy as np
import time
from tkinter import *

# Buat jendela GUI sederhana
window = Tk()

def videoFaceDet():
    # Buka kamera
    video = cv2.VideoCapture(0)

    # Muat Haar Cascade untuk deteksi wajah
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # === Model Age & Gender ===
    ageProto = r"C:\Users\ASUS\Downloads\video-face-detection-master\video-face-detection-master\age_deploy.prototxt"
    ageModel = r"C:\Users\ASUS\Downloads\video-face-detection-master\video-face-detection-master\age_net.caffemodel"
    genderProto = r"C:\Users\ASUS\Downloads\video-face-detection-master\video-face-detection-master\gender_deploy.prototxt"
    genderModel = r"C:\Users\ASUS\Downloads\video-face-detection-master\video-face-detection-master\gender_net.caffemodel"

    # Muat model DNN
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    # Daftar mean values dan label
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-30)', '(31-43)', '(44-60)', '(60-100)']
    genderList = ['Male', 'Female']

    # Loop utama
    while True:
        check, frame = video.read()
        if not check:
            print("Kamera tidak terbaca.")
            break

        grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=5)

        for x, y, w, h in faces:
            # Gambar kotak wajah
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Ambil area wajah
            face = frame[y:y + h, x:x + w].copy()
            if face.size == 0:
                continue

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Prediksi gender
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            # Prediksi usia
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            # === Perkiraan warna kulit (dari rata-rata warna wajah) ===
            mean_color = cv2.mean(cv2.cvtColor(face, cv2.COLOR_BGR2HSV))
            hue = mean_color[0]

            if hue < 15:
                skin_tone = "Fair"
            elif 15 <= hue < 25:
                skin_tone = "Medium"
            else:
                skin_tone = "Tan/Dark"

            # Tampilkan label di atas wajah
            label = f"{gender}, {age}, {skin_tone}"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Tampilkan hasil
        cv2.imshow("Capturing", frame)

        # Tekan 'q' untuk keluar
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Tutup semua jendela
    video.release()
    cv2.destroyAllWindows()

# GUI Button dan Label
b1 = Button(window, text="Start", command=videoFaceDet)
b1.grid(row=0, column=0)

l1 = Label(window, text="Press Q to Stop Capturing")
l1.grid(row=0, column=1)

window.mainloop()
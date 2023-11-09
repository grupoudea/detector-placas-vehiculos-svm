import numpy as np
import cv2 as cv
import joblib

# cargar modelo
loaded_model = joblib.load('modelo_entrenado2.pkl')


# Abre el video
video_capture = cv.VideoCapture('./videos/placas3.avi')

fps = video_capture.get(cv.CAP_PROP_FPS)

window_width, window_height = 64, 128

win_size = (64, 128)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
num_bins = 9

hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

# se usaron solo para generar la imagenes no placas
extension = ".jpg"
contador = 0

# area de interes
min_x = 1084
min_y = 630

limit_x = 1500
limit_y = 1030

v_a = [min_x, min_y]
v_b = [limit_x, min_y]
v_c = [limit_x, limit_y]
v_d = [min_x, limit_y]


# Definir las constantes de rango de color amarillo
lower_yellow = np.array([15, 50, 50], dtype="uint8")
upper_yellow = np.array([35, 255, 255], dtype="uint8")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    pts = np.array([v_a, v_b, v_c, v_d], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    h, w, channel = frame.shape




    frame_gray = frame
    frame_gray_recorte = frame[min_y:limit_y, min_x: limit_x]

    b = np.matrix(frame_gray_recorte[:, :, 0])
    g = np.matrix(frame_gray_recorte[:, :, 1])
    r = np.matrix(frame_gray_recorte[:, :, 2])

    color = cv.absdiff(g, b)

    _, umbral = cv.threshold(color, 40, 255, cv.THRESH_BINARY)

    dilatacion = cv.dilate(umbral, np.ones((3, 3)))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    cierre = cv.morphologyEx(dilatacion, cv.MORPH_CLOSE, kernel)
    cv.imshow('cierre', cierre)

    contorno, _ = cv.findContours(umbral, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contornos = sorted(contorno, key=lambda x: cv.contourArea(x), reverse=True)


    cv.imshow('Video_2_recorte', umbral)
    x_detected, y_detected = 0, 0

    print(f'frame_shape[0] {frame_gray.shape[0]} - [1]={frame_gray.shape[1]}')
    print(f'win[0]={win_size[0]} - win[1]={win_size[1]}')

    for y in range(min_y, limit_y - win_size[1], int(win_size[1] / 2)):
        for x in range(min_x, limit_x - win_size[0], int(win_size[0] / 2)):
            window = frame_gray[y:y + win_size[1], x:x + win_size[0]]

            features = hog.compute(window)
            # window_reshape = features.reshape(1, -1)

            window_reshape = np.vstack([features.T])  # transpuesta

            prediction = loaded_model.predict(window_reshape)
            # print(prediction)
            if prediction == 1:
                # cv.imwrite(f"./no_placas_temp/placa_{contador}_{x}_{y}{extension}", window)
                hsv_roi = cv.cvtColor(window, cv.COLOR_BGR2HSV)
                # cv.rectangle(frame, (x, y), (x + window_width, y + window_height), (0, 50, 255), 2)

                # Verificar si la región tiene colores en el rango de amarillo
                mask = cv.inRange(hsv_roi, lower_yellow, upper_yellow)
                yellow_pixels = cv.countNonZero(mask)

                # Umbral para considerar la región como amarilla
                yellow_threshold = 0. * window.size
                if 1200 <= yellow_pixels < 2100:
                    contador = contador + 1
                    x_detected, y_detected = x, y
                    # cv.imshow('Video_2', window)
                    # cv.imwrite(f"./placas_con_detector_2.0/placa_{contador}_{yellow_pixels}_{x}_{y}{extension}", window)
                    cv.rectangle(frame, (x, y), (x + window_width, y + window_height), (0, 255, 0), 2)

            # cv.imshow('Video2', frame)
            # cv.waitKey(5)

    # cv.rectangle(frame, (x_detected, y_detected), (x_detected + window_width, y_detected + window_height), (0, 255, 0), 2)
    cv.imshow('Video', frame)

    if cv.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()

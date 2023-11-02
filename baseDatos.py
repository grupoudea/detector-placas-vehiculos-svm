import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


def callback_save(event):
    print(f"Entró en x={x1}, y={y1}")
    ancho = x2 - x1
    alto = y2 - y1
    nombre_base = "placa_"
    extension = ".jpg"
    global contador
    if ret:
        roi = cv2.cvtColor(frame[y1:y1 + alto, x1:x1 + ancho], cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{nombre_base}{contador}{extension}", roi)            #Ruta de a guardar la imagen antes de image_frame
        contador = contador + 1

def mouse_callback_left(event):
    global x1, y1
    x1, y1 = event.x, event.y
    print(f"Clic en x={x1}, y={y1}")

def mouse_callback_rigth(event):
    global x2, y2
    x2, y2 = event.x, event.y
    print(f"Clic en x={x2}, y={y2}")

# Función para abrir un cuadro de diálogo y seleccionar un archivo de video
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
    global play
    if file_path:
        play_video(file_path)
        play = True
        print(play)

def update_frame():
    global play
    global cap
    global frame
    global ret

    print('Global_play: ', play)
    if play:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            label.config(image=photo)
            label.image = photo
            label.after(30, update_frame)
        else:
            cap.release()

# Función para reproducir el video seleccionado
def play_video(video_path):
    global cap
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return

    update_frame()

def key_event_callback(event):
    global play
    print("Play: ", play)
    play = not play
    print("not play: ", play)
    if play:
        update_frame()

# Crear una ventana de tkinter
root = tk.Tk()
root.title("Reproducir Video")
root.bind("<space>", key_event_callback)
root.bind("<KeyPress-s>", callback_save)
play = True

# Crear un marco en la ventana
# frame = tk.Canvas(root, width=400, height=300)
# frame.bind("<Button-1>", mouse_callback)  # Bind al evento de clic izquierdo del ratón
# frame.pack()

# Botón para salir
quit_button = tk.Button(root, text="Salir", command=root.destroy)
quit_button.pack()


# Botón para seleccionar un archivo de video
open_button = tk.Button(root, text="Seleccionar Video", command=open_file)
open_button.pack()

# Etiqueta para mostrar el video
label = tk.Label(root)
label.bind("<Button-1>", mouse_callback_left)

label.bind("<Button-3>", mouse_callback_rigth)

# TODO
# Segundo evento para resetear medidas
# Convertir la imagen capturada de RGB a BGR
# números de imagen a hacer por cada persona (rango)

label.pack()
cap = None
frame = None
ret = None
x1 = 0
y1 = 0
x2 = 0
y2 = 0
contador = 4000

root.mainloop()



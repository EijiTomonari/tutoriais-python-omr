import cv2 as cv

# Inicializa a câmera
cap = cv.VideoCapture(1)  # O número seleciona a câmera
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)  # Largura de 1920px
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)  # Altura de 1080px

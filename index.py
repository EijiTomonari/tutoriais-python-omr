import cv2 as cv


def capturaImagem():
    # Inicializa a câmera
    cap = cv.VideoCapture(1)  # O número seleciona a câmera
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)  # Largura de 1920px
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)  # Altura de 1080px

    if not cap.isOpened():  # Verifica se a câmera foi inicializada
        return None

    for i in range(0, 10):  # Lê 10 frames
        ret, frame = cap.read()
    cv.imshow("Imagem capturada", frame)  # Mostra a imagem capturada
    cv.waitKey(0)  # Aguarda infinitamente por uma tecla pressionada
    return frame

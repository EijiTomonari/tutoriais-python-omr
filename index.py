import cv2 as cv
import numpy as np

LARGURA = 1920
ALTURA = 1080
QUESTOES = 5
ALTERNATIVAS = 5


def capturaImagem():
    # Inicializa a câmera
    cap = cv.VideoCapture(1)  # O número seleciona a câmera
    cap.set(cv.CAP_PROP_FRAME_WIDTH, LARGURA)  # Define a largura da captura
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, ALTURA)  # Define a altura da captura

    if not cap.isOpened():  # Verifica se a câmera foi inicializada
        return None

    ret, frame = cap.read()  # Lê um frame
    # Gira a imagem 90 graus no sentido anti-horário
    frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)

    return frame


def detectaContornos(img):
    # Converte a imagem para preto e branco
    imgCinza = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Aplica um desfoque gaussiano à imagem
    imgDesfoque = cv.GaussianBlur(imgCinza, (11, 11), 1)
    # Converte a imagem para binária usando um limite
    imgRet, imgLimite = cv.threshold(imgDesfoque, 127, 255, 1)
    kernel = np.ones((5, 5))
    # Aplica uma dilatação para aumentar a espessura dos contornos
    imgLimite = cv.dilate(imgLimite, kernel, iterations=2)
    # Transforma os contornos em um array a partir da imagem binária
    contornos, hierarquia = cv.findContours(
        imgLimite, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Faz uma cópia da imagem original
    imgContornos = img.copy()
    # Desenha os contornos na imagem
    cv.drawContours(imgContornos, contornos, -1, (0, 255, 0), 5)
    return contornos, imgContornos, imgLimite


def encontraMaiorContorno(contornos, img):
    # Encontra o maior contorno
    maiorContorno = max(contornos, key=len)
    # Faz uma cópia da imagem original para modificações
    imgRetangulo = img.copy()
    # Desenha o maior contorno
    imgMaiorContorno = cv.drawContours(
        imgRetangulo, maiorContorno, -1, (0, 255, 0), 10)
    return maiorContorno, imgMaiorContorno


def reordenarPontos(pontos):

    pontos = pontos.reshape((4, 2))  # Formata o array
    pontosOrdenados = np.zeros(
        (4, 1, 2), dtype=np.int32)  # Cria um array vazio
    soma = pontos.sum(1)  # Soma as coordenadas de cada ponto
    diferenca = np.diff(pontos, axis=1)  # Subtrai as coordenadas de cada ponto

    # Usa a soma e a diferenca dos pontos para colocá-los na ordem certa
    pontosOrdenados[0] = pontos[np.argmin(soma)]
    pontosOrdenados[1] = pontos[np.argmin(diferenca)]
    pontosOrdenados[2] = pontos[np.argmax(diferenca)]
    pontosOrdenados[3] = pontos[np.argmax(soma)]

    return pontosOrdenados


def corrigePerspectiva(maiorContorno, img):
    fatorCorrecao = 0.7  # Fator de correção para a altura da imagem final
    # Calcula o perímetro do maior contorno
    perimetro = cv.arcLength(maiorContorno, True)
    # Simplifica o número de vértices do maior contorno
    simplificado = cv.approxPolyDP(maiorContorno, 0.02 * perimetro, True)
    # Coloca os vértices do contorno na ordem correta
    simplificado = reordenarPontos(simplificado)
    pts1 = np.float32(simplificado)  # Ponto inicial da imagem
    pts2 = np.float32([[0, 0], [ALTURA, 0], [0, round(LARGURA*fatorCorrecao)], [
                      ALTURA, round(LARGURA*fatorCorrecao)]])  # Ponto final da imagem
    # Calcula a perspectiva usando os dois pontos
    matriz = cv.getPerspectiveTransform(pts1, pts2)
    imgPlanificada = cv.warpPerspective(
        img, matriz, (ALTURA, round(LARGURA*fatorCorrecao)))  # Aplica a correção de perspectiva
    return imgPlanificada


def cortaAreaDasLacunas(img):
    imgAreaLacunas = img.copy()
    # Corta somente a área das lacunas
    imgAreaLacunas = imgAreaLacunas[340:1300, 40:1040]
    return imgAreaLacunas


def identificaLacunas(contornos, imgPlanificada):
    imgLacunas = imgPlanificada.copy()
    proporcaoMinima = 0.9
    proporcaoMaxima = 1.1
    tamanhoMaximo = 70
    tamanhoMinimo = 50
    contornosLacunas = []
    for contorno in contornos:
        (x, y, largura, altura) = cv.boundingRect(contorno)
        proporcao = largura / altura
        if largura <= tamanhoMaximo and largura >= tamanhoMinimo and altura <= tamanhoMaximo and altura >= tamanhoMinimo and proporcao >= proporcaoMinima and proporcao <= proporcaoMaxima:
            contornosLacunas.append(contorno)
    for lacuna in contornosLacunas:
        (x, y, largura, altura) = cv.boundingRect(lacuna)
        cv.rectangle(imgLacunas, (x, y), (x + largura, y + altura),
                     (0, 255, 63), 5)
    return contornosLacunas, imgLacunas


def ordenarLacunas(contornosLacunas, metodo="esq-dir"):
    indice = 0
    if (metodo == "cim-bai"):
        indice = 1
    lacunas = [cv.boundingRect(lacuna) for lacuna in contornosLacunas]
    (contornosLacunas, lacunas) = zip(*sorted(zip(contornosLacunas, lacunas),
                                              key=lambda b: b[1][indice], reverse=False))
    return (contornosLacunas, lacunas)


def identificaMarcacoes(contornosLacunas, imgAreaLacunas, imgLimite):
    imgLacunasLinhas = imgAreaLacunas.copy()
    imgMarcacoes = imgAreaLacunas.copy()
    imgLimiteCopia = imgLimite.copy()
    # Array opcional de cores para as linhas
    cores = [(255, 0, 0), (0, 0, 255),
             (89, 255, 0), (0, 242, 255), (221, 0, 255)]
    valoresPixels = np.zeros((QUESTOES, ALTERNATIVAS), dtype=int)
    for (indice, linha) in enumerate(np.arange(0, len(contornosLacunas), ALTERNATIVAS)):
        contornosLinhas = ordenarLacunas(
            contornosLacunas[linha:linha + ALTERNATIVAS])[0]

        # Loop opcional para desenhar as cores das linhas
        for linha in contornosLinhas:
            x, y, w, h = cv.boundingRect(linha)
            cv.rectangle(imgLacunasLinhas, (x, y), (x + w, y + h),
                         cores[indice], -1)

        for (j, linha) in enumerate(contornosLinhas):
            # Constrói uma máscara
            mask = np.zeros(imgLimiteCopia.shape, dtype="uint8")
            # Mostra apenas a lacuna sendo analisada
            cv.drawContours(mask, [linha], -1, 255, -1)
            # Aplica a máscara à imagem binária
            mask = cv.bitwise_and(imgLimiteCopia, imgLimiteCopia, mask=mask)
            # Conta o número de pixels não-nulos na lacuna
            total = cv.countNonZero(mask)
            # Insere o valor de pixels na matriz
            valoresPixels[indice][j] = total
            # if the current total has a larger number of total
            # non-zero pixels, then we are examining the currently
            # checked box
    media = np.average(valoresPixels)
    desvioPadrao = np.std(valoresPixels)
    limitePixels = media + (desvioPadrao*1.7)
    for (indice, linha) in enumerate(np.arange(0, len(contornosLacunas), ALTERNATIVAS)):
        contornosLinhas = ordenarLacunas(
            contornosLacunas[linha:linha + ALTERNATIVAS])[0]
        for (j, lacuna) in enumerate(contornosLinhas):
            if(valoresPixels[indice][j] >= limitePixels):
                cv.drawContours(imgMarcacoes, [lacuna], -1, (0, 0, 255), 10)
    return imgLacunasLinhas, imgMarcacoes


def main():
    img = capturaImagem()
    # img = cv.imread("exemplo2.jpg") # Linha opcional para usar imagens de exemplo ao invés da função de leitura
    contornos, imgContornos, imgLimite = detectaContornos(img)
    maiorContorno, imgMaiorContorno = encontraMaiorContorno(contornos, img)
    imgPlanificada = corrigePerspectiva(maiorContorno, img)
    imgAreaLacunas = cortaAreaDasLacunas(imgPlanificada)
    contornosPlanificados, imgContornosPlanificados, imgLimitePlanificado = detectaContornos(
        imgAreaLacunas)
    contornosLacunas, imgLacunas = identificaLacunas(
        contornosPlanificados, imgAreaLacunas)
    contornosLacunas = ordenarLacunas(contornosLacunas, metodo='cim-bai')[0]
    imgLacunasLinhas, imgMarcacoes = identificaMarcacoes(
        contornosLacunas, imgAreaLacunas, imgLimitePlanificado)

    # Mostra cada uma das imagens (aguarda uma tecla ser pressionada para mostrar a próxima)
    cv.imshow("Imagem Original", img)
    cv.waitKey(0)
    cv.imshow("Contornos", imgContornos)
    cv.waitKey(0)
    cv.imshow("Maior Contorno", imgMaiorContorno)
    cv.waitKey(0)
    cv.imshow("Imagem Planificada", imgPlanificada)
    cv.waitKey(0)
    cv.imshow("Contornos Planificados", imgContornosPlanificados)
    cv.waitKey(0)
    cv.imshow("Imagem Lacunas", imgLacunas)
    cv.waitKey(0)
    cv.imshow("Linhas ordenadas", imgLacunasLinhas)
    cv.waitKey(0)
    cv.imshow("Marcadas", imgMarcacoes)
    cv.waitKey(0)


if __name__ == '__main__':
    main()

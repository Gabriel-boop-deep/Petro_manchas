import cv2
import tensorflow as tf
import numpy as np

# Carrega o modelo treinado
model = tf.keras.models.load_model('modelo.h5')

# Define uma função para processar a imagem e detectar as manchas de petróleo
def detecta_manchas(imagem):
    # Converte a imagem para escala de cinza
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    # Aplica um filtro de suavização para reduzir o ruído
    suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)
    # Detecta as bordas na imagem usando o algoritmo Canny
    bordas = cv2.Canny(suavizada, 30, 100)
    # Encontra os contornos na imagem
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Cria uma cópia da imagem para desenhar as manchas encontradas
    imagem_com_manchas = imagem.copy()
    # Para cada contorno encontrado
    for contorno in contornos:
        # Calcula a área do contorno
        area = cv2.contourArea(contorno)
        # Se a área for maior que 1000 pixels
        if area > 1000:
            # Extrai as coordenadas do retângulo que contém o contorno
            x, y, w, h = cv2.boundingRect(contorno)
            # Extrai a região da imagem correspondente ao retângulo
            regiao = imagem[y:y+h, x:x+w]
            # Redimensiona a região para o tamanho esperado pelo modelo
            regiao_redimensionada = cv2.resize(regiao, (224, 224))
            # Normaliza os valores dos pixels para o intervalo [0, 1]
            regiao_normalizada = regiao_redimensionada / 255.0
            # Adiciona uma dimensão extra para indicar o número de amostras
            regiao_amostras = np.expand_dims(regiao_normalizada, axis=0)
            # Faz a predição do modelo
            predicao = model.predict(regiao_amostras)[0]
            # Se a predição indicar que a região contém uma mancha de petróleo
            if predicao > 0.5:
                # Desenha um retângulo na imagem com a região
                cv2.rectangle(imagem_com_manchas, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # Retorna a imagem com as manchas identificadas
    return imagem_com_manchas

# Carrega a imagem que queremos analisar
imagem = cv2.imread('imagem.jpg')
# Chama a função

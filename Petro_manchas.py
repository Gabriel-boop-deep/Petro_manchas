import cv2
import numpy as np

# Carregando a imagem de satélite
img = cv2.imread('imagem_satelite.jpg')

# Convertendo a imagem para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicando um filtro Gaussiano para reduzir o ruído
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Aplicando a técnica de thresholding para separar as regiões escuras das claras
ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# Encontrando os contornos das regiões escuras
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Desenhando os contornos encontrados na imagem original
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

# Exibindo a imagem resultante
cv2.imshow('Imagem', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

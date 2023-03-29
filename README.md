# Identificação de manchas de Petróleo

Um parágrafo da descrição do projeto vai aqui

## 🚀 Começando

O código apresentado tem como objetivo identificar manchas de petróleo no mar a partir de imagens utilizando técnicas de processamento de imagem e aprendizado de máquina.

### 📋 Pré-requisitos

Para começar rodar o código, devemos saber conceitos como aprendizado de máquina e a linguagem Python


### 🔧 Instalação

Para identificar manchas de petróleo no mar a partir de imagens, podemos utilizar técnicas de processamento de imagem e aprendizado de máquina. Vou apresentar um exemplo de código em Python que utiliza a biblioteca OpenCV para processar as imagens e a biblioteca TensorFlow para treinar um modelo de aprendizado de máquina.


## ⚙️ Executando os testes

Antes de começar, é importante notar que este código é apenas um exemplo e pode ser adaptado para diferentes situações. Além disso, é importante ter um conjunto de imagens rotuladas para treinar o modelo. Neste exemplo, vou supor que já temos esse conjunto de imagens.

### 🔩 Analise os testes de ponta a ponta

O código apresentado tem como objetivo identificar manchas de petróleo no mar a partir de imagens utilizando técnicas de processamento de imagem e aprendizado de máquina.

```
import cv2
import tensorflow as tf
import numpy as np

# Carrega o modelo treinado
model = tf.keras.models.load_model('modelo.h5')
```

### ⌨️ E testes de estilo de codificação

Explique que eles verificam esses testes e porquê.

```

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
```

## 📦 Implantação

Primeiramente, o código carrega o modelo treinado utilizando a biblioteca TensorFlow. Esse modelo deve ter sido treinado previamente com um conjunto de imagens rotuladas, onde cada imagem é classificada como tendo ou não tendo manchas de petróleo.

Em seguida, é definida uma função chamada "detecta_manchas", que recebe como parâmetro uma imagem e retorna a mesma imagem com as manchas de petróleo identificadas. Essa função utiliza a biblioteca OpenCV para processar a imagem e identificar as manchas.

Dentro da função "detecta_manchas", a imagem é convertida para escala de cinza e é aplicado um filtro de suavização para reduzir o ruído. Em seguida, são detectadas as bordas na imagem utilizando o algoritmo Canny e são encontrados os contornos na imagem.

Para cada contorno encontrado, a função verifica se a área do contorno é maior que 1000 pixels. Se for, a função extrai a região da imagem correspondente ao contorno, redimensiona essa região para o tamanho esperado pelo modelo e normaliza os valores dos pixels para o intervalo [0, 1].

A região normalizada é então passada como entrada para o modelo, que faz a predição e retorna um valor entre 0 e 1 indicando a probabilidade de a região conter uma mancha de petróleo. Se a predição for maior que 0.5, a função desenha um retângulo na imagem com a região correspondente ao contorno, indicando que aquela região contém uma mancha de petróleo.

Por fim, o código carrega a imagem que deseja-se analisar e chama a função "detecta_manchas" passando essa imagem como parâmetro. O resultado é a mesma imagem com as manchas de petróleo identificadas.


## 🛠️ Construído com

Mencione as ferramentas que você usou para criar seu projeto
S

## 🖇️ Colaborando

Por favor, leia o [COLABORACAO.md](https://gist.github.com/usuario/linkParaInfoSobreContribuicoes) para obter detalhes sobre o nosso código de conduta e o processo para nos enviar pedidos de solicitação.

## 📌 Versão

Nós usamos [SemVer](http://semver.org/) para controle de versão. Para as versões disponíveis, observe as [tags neste repositório](https://github.com/suas/tags/do/projeto). 

## ✒️ Autores

Mencione todos aqueles que ajudaram a levantar o projeto desde o seu início

* **Um desenvolvedor** - *Trabalho Inicial* - [umdesenvolvedor](https://github.com/linkParaPerfil)
* **Fulano De Tal** - *Documentação* - [fulanodetal](https://github.com/linkParaPerfil)

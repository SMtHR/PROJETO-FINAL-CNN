import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow import keras

DATADIR = "Datasets\IdentificandoPessoas"
CATEGORIAS = ["Pessoas", "SemPessoas"]

imagens_treinamento = []
classes_treinamento = []
tamanho_imagem = 110

def carregamento_e_pre_processamento():
  for categoria in CATEGORIAS:
    path = os.path.join(DATADIR, categoria)
    class_num = CATEGORIAS.index(categoria)
    for arquivo in os.listdir(path):
      img_path = os.path.join(path, arquivo)
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Carregando imagem e transformando-a para escala de CINZA
      img = cv2.resize(img, (tamanho_imagem, tamanho_imagem)) # Redimensionando a imagem
      img = cv2.fastNlMeansDenoising(img, h=10) # Reduzindo ruído da imagem

      kernel_sharpening = np.array([[-1,-1,-1], # Matriz de nitidez a ser aplicada à imagem
                                   [-1,9,-1],
                                   [-1,-1,-1]])
      img = cv2.filter2D(img, -1, kernel_sharpening) # Aplicando a matriz à imagem para aumentar sua nitidez

      imagens_treinamento.append(img)
      classes_treinamento.append(class_num)

  return np.array(imagens_treinamento), np.array(classes_treinamento)

imagens_treinamento, classes_treinamento = carregamento_e_pre_processamento()

plt.imshow(imagens_treinamento[3], cmap="gray")
plt.show()
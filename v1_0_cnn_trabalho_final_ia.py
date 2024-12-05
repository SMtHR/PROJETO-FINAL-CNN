"""

Original file is located at
    https://colab.research.google.com/drive/1PV9CCfacLxGkCqeS65vSmfSk81PSkm0p

# **MOVIFLOW - Solução para Transporte Público**

---


### **Utilizando Redes Neurais Convolucionais (CNN) para reconhecimento de pessoas em imagens**
* *O objetivo do algoritmo é conseguir estabelecer se uma imagem **contém** uma pessoa ou não. Essa premissa seria utilizada em um sistema de **reconhecimento em tempo real** que determinaria a **quantidade de pessoas aguardando** em uma parada de transporte público.*

Bibliotecas usadas:
TensorFlow, OpenCV, MatPlotLib, Pickle

Integrantes:

*   Artur Aquino Lacerda – 324219992
*   Diego Manini – 32424894
*   Ian Jhonatan Oliveira de Pinho – 322130340
*   Matheus Silva Rodrigues de Souza – 324142016
*   Nícolas Alves Pinheiro Chagas – 322126250

## **Imports, Constantes e exemplo de imagem a ser processada**
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle
import matplotlib.pyplot as plt
import os
import cv2

NAME = "Identificando-pessoas-cnn-{}".format(int(time.time()))
TENSORBOARD = TensorBoard(log_dir='logs/{}'.format(NAME))

DATADIR = "Datasets\IdentificandoPessoas"
CATEGORIAS = ["Pessoas", "SemPessoas"]

for categoria in CATEGORIAS:
    path = os.path.join(DATADIR, categoria)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        plt.imshow(img_array)
        plt.show()
        break
    break

"""## **Incorporando o algoritmo de preprocessamento de imagens ao modelo**


"""

imagens_treinamento = []
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

      imagens_treinamento.append([img, class_num])

carregamento_e_pre_processamento()

"""## **Número de Imagens processadas pelo algoritmo de preprocessamento**"""

print(len(imagens_treinamento))

"""## **Embaralhando imagens antes do treinamento**"""

import random
random.shuffle(imagens_treinamento)

feature_set = []
label_set = []

for features, label in imagens_treinamento:
  feature_set.append(features)
  label_set.append(label)

feature_set = np.array(feature_set).reshape(-1, tamanho_imagem, tamanho_imagem, 1)
label_set = np.array(label_set)

"""## **Como o modelo verá a imagem**"""

plt.imshow(feature_set[0], cmap='gray')
plt.show()

#Salvando a feature set e label set
pickle_out = open("feature_set.pickle", "wb")
pickle.dump(feature_set, pickle_out)
pickle_out.close()

pickle_out = open("label_set.pickle", "wb")
pickle.dump(label_set, pickle_out)
pickle_out.close()

pickle_in = open("feature_set.pickle", "rb")
feature_set = pickle.load(pickle_in)

print("Feature set shape:", feature_set.shape)
print("Label set shape:", label_set.shape)

"""## **Algoritmo do modelo de classificação de imagens**"""

#Normalizando imagens antes de alimentar o algoritmo
feature_set = tf.keras.utils.normalize(feature_set, axis=1)

#Camada de entrada (Input)
model = Sequential()
model.add(tf.keras.Input(feature_set.shape[1:]))

#Camada escondidas (Hidden Layers)
model.add(Conv2D(64, (3,3))) #Convolucional
model.add(Activation("relu")) #Ativação
model.add(MaxPooling2D(pool_size=(2,2))) #Pooling

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #Achatamento da matriz em vetor

#Camada completamente conectada (Fully Connected Layer)
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.3))

#Camada de saída (Output)
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(feature_set, label_set, batch_size=32, epochs=8, validation_split=0.1, callbacks=[TENSORBOARD])

model.save('CNN.model.keras')

"""## **Testando a classificação de imagens externas**"""

def preparar_imagem(filepath):
  img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
  new_array = cv2.resize(img_array, (tamanho_imagem, tamanho_imagem))
  return new_array.reshape(-1, tamanho_imagem, tamanho_imagem, 1)

prediction = model.predict([preparar_imagem("teste.jpeg")])
print(CATEGORIAS[int(prediction[0][0])])

prediction = model.predict([preparar_imagem("teste2.jpg")])
print(CATEGORIAS[int(prediction[0][0])])
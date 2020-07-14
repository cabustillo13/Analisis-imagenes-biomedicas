"""Nota: La mascara mask debe tener las mismas dimensiones que la imagen a analizar"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import scipy.ndimage as ndi

# Imagen a analizar
image = cv2.imread('./Imagenes/resonancia0.jpg', cv2.IMREAD_GRAYSCALE)
filtro= cv2.GaussianBlur(image,(5, 5), 2)
mask = filtro > 150

# El algoritmo de etiquetado trata los valores 0 como pixeles de fondo, y luego busca todos los objetos separados en el fondo
labels, nlabels = ndi.label(mask)

plt.imshow(labels, cmap = "rainbow")
plt.axis("off")
plt.show()

# Seleccion de etiqueta individual
label1 = np.where(labels==1,image,0) #Donde las etiquetas valen 1 devuelve el valor de image, sino devuelve un cero
plt.imshow(label1, cmap = "rainbow")
plt.axis("off")
plt.show()

# Seleccionar varias etiquetas en la imagen
label2 = np.where(labels < 3,image,0)
plt.imshow(label2, cmap = "rainbow")
plt.axis("off")
plt.show()

# Cuadro delimitador o bounding box
cajas = ndi.find_objects(labels)

f, axes = plt.subplots(1, 4, figsize=(16, 5))
axes[0].imshow(image[cajas[0]])
axes[0].set_title('cajas[0]')
axes[1].imshow(image[cajas[1]])
axes[1].set_title('cajas[1]')
axes[2].imshow(image[cajas[2]])
axes[2].set_title('cajas[2]')
axes[3].imshow(image[cajas[3]])
axes[3].set_title('cajas[3]')

for ax in axes:
    ax.axis('off')

plt.show()

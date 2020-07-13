"""Nota: La mascara mask debe tener las mismas dimensiones que la imagen a analizar"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

# Imagen a analizar
image = cv2.imread('./Imagenes/radiografia.jpg', cv2.IMREAD_GRAYSCALE)

# Histograma de intensidad de la imagen en escala de grises
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# Ver Histograma 
plt.plot(hist, color='gray' )
plt.xlabel('Intensidad de iluminacion')
plt.ylabel('Cantidad de pixeles')
plt.grid()
plt.show()

#Enmascaramiento
#Para definir el valor de mask1 tenemos que ver en el grafico del histograma donde se presenta una fuerte caida en la intensidad de iluminacion
mask1 = image > 19          #Resaltar objeto
mask2 = image > 108         #Resaltar hueso
mask3 = mask1 & ~mask2      #Resaltar tejido

f, axes = plt.subplots(1, 4, figsize=(16, 5))
axes[0].imshow(image)
axes[0].set_title('Original')
axes[1].imshow(mask1)
axes[1].set_title('Resaltar objeto')
axes[2].imshow(mask2)
axes[2].set_title('Resaltar hueso')
axes[3].imshow(mask3)
axes[3].set_title('Resaltar tejido')


# En estos analisis biomedicos muchas veces no se muestran los ejes y ticks
for ax in axes:
    ax.axis('off')

plt.show()

# Las mascaras tambien se pueden utilizar para filtrar, por ejemplo filtrar objetos que no son parte del hueso (marcas de agua, objetos extranos, etc.)
im_hueso = np.where(image > 108,image,0) # Cuando image > 108 devuelve image, sino devuelve un cero

"""Nota: Los datos son ruidosos, por lo que sus mascaras rara vez son perfectas"""
"""Una solucion es aumentar el tamano de la mascara, puede agregar pixeles alrededos, a esto se le llama Dilatacion"""
kernel = np.ones((5,5),np.uint8)
im_dilatada = cv2.dilate(im_hueso, kernel, iterations = 5)
"""Esto tambien sirve para no enmascara accidentalmente los pixeles que realmente interesan"""

# Erosion (Metodo opuesto a lo que realiza la dilatacion)
kernel = np.ones((5,5),np.uint8)
im_erosionada = cv2.erode(im_hueso, kernel, iterations = 5)

f, axes = plt.subplots(1, 4, figsize=(16, 5))
axes[0].imshow(image)
axes[0].set_title('Original')
axes[1].imshow(im_hueso)
axes[1].set_title('Filtrado con enmascaramiento')
axes[2].imshow(im_dilatada)
axes[2].set_title('Imagen dilatada')
axes[3].imshow(im_erosionada)
axes[3].set_title('Imagen erosionada')

for ax in axes:
    ax.axis('off')
plt.show()

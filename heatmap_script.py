import cv2
import numpy as np
import requests
from matplotlib import pyplot as plt


def generate_attention_map(frame):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar el algoritmo de Shi-Tomasi para detectar esquinas
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

    # Crear una imagen en blanco para el mapa de atención
    attention_map = np.zeros_like(frame, dtype=np.uint8)

    # Iterar sobre las esquinas detectadas y dibujar círculos en el mapa de atención
    if corners is not None:
        corners = corners.astype(int)
        center_x = frame.shape[1] / 2
        center_y = frame.shape[0] / 2
        high_attention_limit = min(frame.shape[:2]) / 3
        medium_attention_limit = min(frame.shape[:2]) * 2 / 3

        for corner in corners:
            x, y = corner.ravel()
            # Definir el radio del círculo basado en la importancia del punto
            radius = 50
            color = (255, 0, 0)  # Rojo por defecto
            # Calcular la distancia al centro, considerado el punto de mayor atención
            distance = np.hypot(center_x - x, center_y - y)
            # Clasificar los puntos en tres niveles de atención según su distancia al centro de la imagen
            if distance < high_attention_limit:  # Puntos de interés alto (rojo)
                color = (0, 0, 255)
            elif distance < medium_attention_limit:  # Puntos de interés medio (verde)
                color = (0, 255, 0)  # Verde
            # Dibujar el círculo
            cv2.circle(attention_map, (x, y), radius, color, -1)

    # Aplicar un desenfoque gaussiano al mapa de atención para suavizar los bordes
    attention_map = cv2.GaussianBlur(attention_map, (75, 75), 0)

    return attention_map


def main():
    # Descargar la imagen desde el enlace
    image_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo decodificar la imagen descargada")

    # Generar el mapa de atención
    attention_map = generate_attention_map(image)

    output_heatmap_path = 'imagen_con_mapa_de_atencion.png'
    cv2.imwrite(output_heatmap_path, attention_map)

    # Superponer la imagen original con el mapa de atención
    alpha = 0.6  # Factor de transparencia
    result = cv2.addWeighted(image, alpha, attention_map, 1 - alpha, 0)
    cv2.imwrite('resultado_mapa_de_atencion.png', result)

    # Mostrar la imagen resultante
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Imagen con Mapa de Atención')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()

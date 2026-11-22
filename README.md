# Mapa de atención visual con OpenCV

Prototipo de visión por computadora que detecta puntos de interés en una imagen mediante el algoritmo de Shi-Tomasi y genera un mapa de calor según su distancia al centro.

> Este proyecto utiliza técnicas clásicas de visión por computadora; no entrena ni ejecuta un modelo de inteligencia artificial.

## Resultado

![Mapa de atención superpuesto sobre la imagen](resultado_mapa_de_atencion.png)

![Mapa de atención generado](imagen_con_mapa_de_atencion.png)

Los puntos detectados se agrupan en tres niveles de atención y se suavizan con un filtro gaussiano. El programa genera:

- `imagen_con_mapa_de_atencion.png`: mapa de atención aislado.
- `resultado_mapa_de_atencion.png`: mapa superpuesto sobre la imagen original.

## Tecnologías

- Python
- OpenCV
- NumPy
- Matplotlib
- Requests

## Instalación

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Uso

```bash
python heatmap_script.py
```

El ejemplo descarga una imagen de muestra del repositorio oficial de OpenCV, procesa sus puntos de interés, guarda las imágenes resultantes y muestra la superposición en pantalla.

## Cómo funciona

1. Convierte la imagen a escala de grises.
2. Detecta esquinas con `cv2.goodFeaturesToTrack`.
3. Clasifica los puntos por distancia al centro de la imagen.
4. Dibuja y suaviza las zonas para formar el mapa de atención.
5. Combina el mapa con la imagen original.

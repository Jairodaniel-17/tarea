import streamlit as st
import numpy as np
from PIL import Image, ImageFilter

# Configuración de la página
st.set_page_config(
    page_title="Tarea 2",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Barra lateral para cargar una imagen y seleccionar un filtro
imagen_original = st.sidebar.file_uploader(
    label="Sube una imagen para aplicar algún filtro", type=["png", "jpg", "jpeg"]
)
filtros = [
    "Prewitt",
    "Sobel",
    "Roberts",
    "Promedio",
    "Gaussiano",
    "Identidad",
    "inverso",
    "binarizado",
    "binarizado_inv",
    "canny",
]
seleccion = st.sidebar.selectbox("Selecciona un filtro", filtros)


# Funciones de filtro
def aplicar_filtro_prewitt(imagen):
    return imagen.filter(ImageFilter.FIND_EDGES)


def aplicar_filtro_sobel(imagen):
    return imagen.filter(ImageFilter.FIND_EDGES)


def aplicar_filtro_roberts(imagen):
    return imagen.filter(ImageFilter.CONTOUR)


def aplicar_filtro_promedio(imagen):
    return imagen.filter(ImageFilter.SMOOTH_MORE)


def aplicar_filtro_gaussiano(imagen):
    return imagen.filter(ImageFilter.GaussianBlur(radius=2))


def aplicar_filtro_identidad(imagen):
    return imagen


def inverso(P):
    Q = 255 - P
    return Q


def binarizado(P, u):
    Q = (P >= u).astype(int)
    return Q


def binarizado_inv(P, u):
    Q = (P < u).astype(int)
    return Q


def aplicar_filtro_canny(imgNP):
    imgNP = imgNP.convert("L")
    gradiente_x = np.gradient(imgNP, axis=1)
    gradiente_y = np.gradient(imgNP, axis=0)
    gradiente_x_abs = np.abs(gradiente_x)
    gradiente_y_abs = np.abs(gradiente_y)
    gradiente_magnitud = np.sqrt(gradiente_x_abs**2 + gradiente_y_abs**2)
    umbral_min = 10
    umbral_max = 255
    bordes = (
        (gradiente_magnitud >= umbral_min) & (gradiente_magnitud <= umbral_max)
    ).astype(int)
    imagen_bordes = Image.fromarray(np.uint8(bordes) * 255)
    return imagen_bordes


# Procesamiento de la imagen
def procesar_imagen(imagen_original, seleccion):
    if imagen_original is not None:
        try:
            imgPIL = Image.open(imagen_original)

            if seleccion == "Prewitt":
                resultado = aplicar_filtro_prewitt(imgPIL)
            elif seleccion == "Sobel":
                resultado = aplicar_filtro_sobel(imgPIL)
            elif seleccion == "Roberts":
                resultado = aplicar_filtro_roberts(imgPIL)
            elif seleccion == "Promedio":
                resultado = aplicar_filtro_promedio(imgPIL)
            elif seleccion == "Gaussiano":
                resultado = aplicar_filtro_gaussiano(imgPIL)
            elif seleccion == "Identidad":
                resultado = aplicar_filtro_identidad(imgPIL)
            elif seleccion == "inverso":
                imgGray = Image.open(imagen_original).convert("L")
                imgNP = np.array(imgGray)
                resultado = inverso(imgNP)
                resultado = Image.fromarray(resultado)
            elif seleccion == "binarizado":
                imgGray = Image.open(imagen_original).convert("L")
                imgNP = np.array(imgGray)
                resultado = binarizado(imgNP, 128)
                resultado = Image.fromarray((resultado * 255).astype(np.uint8))
                resultado.save("binarizada.jpg", "JPEG")
            elif seleccion == "binarizado_inv":
                imgGray = Image.open(imagen_original).convert("L")
                imgNP = np.array(imgGray)
                resultado = binarizado_inv(imgNP, 128)
                resultado = Image.fromarray((resultado * 255).astype(np.uint8))
                resultado.save("binarizada_inv.jpg", "JPEG")
            elif seleccion == "canny":
                resultado = aplicar_filtro_canny(imgPIL)

            col1, col2 = st.columns(2)
            with col1:
                st.image(imgPIL, caption="Imagen Original", use_column_width=True)
            with col2:
                st.image(resultado, caption="Imagen Procesada", use_column_width=True)
        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")


if __name__ == "__main__":
    procesar_imagen(imagen_original, seleccion)

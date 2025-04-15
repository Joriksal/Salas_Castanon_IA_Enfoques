import numpy as np
from hmmlearn import hmm  # Biblioteca para trabajar con modelos ocultos de Markov
from python_speech_features import mfcc  # Para extraer características MFCC de audio
import scipy.io.wavfile as wav  # Para leer archivos de audio WAV
import os  # Para manejar rutas de archivos

# Ruta absoluta donde se encuentran los archivos de audio (modifica según tu sistema)
audio_dir = r"C:\Users\rikis\OneDrive\Documentos\GitHub\Salas_Castanon_IA_Enfoques\02_Probabilidad\audios"

# 1. Extracción de características (MFCC)
# Esta función toma un archivo de audio y extrae sus características MFCC
def extract_features(audio_file):
    # Leer el archivo de audio (frecuencia de muestreo y señal)
    sample_rate, signal = wav.read(audio_file)
    # Calcular las características MFCC
    mfcc_features = mfcc(signal, samplerate=sample_rate, nfft=2048)
    return mfcc_features

# Ejemplo con archivos de entrenamiento
# Extraemos las características MFCC de los archivos de audio "hola.wav" y "adios.wav"
features_hola = extract_features(os.path.join(audio_dir, "hola.wav"))
features_adios = extract_features(os.path.join(audio_dir, "adios.wav"))

# 2. Creación de modelos HMM para cada palabra
# Creamos un modelo HMM para la palabra "hola" con 5 estados y 3 mezclas gaussianas
model_hola = hmm.GMMHMM(n_components=5, n_mix=3, covariance_type="diag")
# Creamos un modelo HMM para la palabra "adios" con 6 estados y 3 mezclas gaussianas
model_adios = hmm.GMMHMM(n_components=6, n_mix=3, covariance_type="diag")

# 3. Entrenamiento (Ajuste de Baum-Welch)
# Entrenamos el modelo "hola" con las características extraídas de "hola.wav"
model_hola.fit(features_hola)
# Entrenamos el modelo "adios" con las características extraídas de "adios.wav"
model_adios.fit(features_adios)

# 4. Reconocimiento de nueva muestra
# Extraemos las características MFCC de un nuevo archivo de audio "nueva_muestra.wav"
nueva_muestra = extract_features(os.path.join(audio_dir, "nueva_muestra.wav"))

# Calculamos la puntuación (log-likelihood) de la nueva muestra para cada modelo
score_hola = model_hola.score(nueva_muestra)
score_adios = model_adios.score(nueva_muestra)

# Comparamos las puntuaciones para determinar qué palabra se reconoce
palabra_reconocida = "hola" if score_hola > score_adios else "adios"
print(f"Palabra reconocida: {palabra_reconocida}")
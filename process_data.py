import os
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import savgol_filter

def loadEMGs(ruta: str) -> Tuple[np.ndarray, np.ndarray]:
    emgs = []
    for _, data in enumerate(os.listdir(ruta)):
        with open(os.path.join(ruta, data), 'rb') as f:
            emgN = [int(val) for val in f.readline().decode().split(',')]
            emgs.append(emgN)
    Fs = 200
    T = 120
    t = np.linspace(0, T, Fs * T)
    return np.array(emgs), t

def plot_emg_window(emg_,n,dV,smooth_window=150,order=2,int_freqs=False):
    p_inicio = int(n*dV)
    p_final  = int((n+1)*dV)
    Fs = 200

    windowsize = 2

    ventana = emg_[p_inicio:p_final]
    emg_fft_abs = abs(fft(ventana))

    df=pd.DataFrame(data={"y":emg_fft_abs})
    df["y_upperEnv"]=df["y"].rolling(window=windowsize).max().shift(int(-windowsize/2))
    envolvente_superior = df['y_upperEnv'].to_numpy()
    suavizada = savgol_filter(envolvente_superior, smooth_window, order)
    suavizada = suavizada[:int(len(suavizada)/2)]

    freq = np.arange(0,len(suavizada))/dV*Fs

    if int_freqs:
      freq, indx = np.unique(np.floor(freq), return_index=True)
      suavizada = suavizada[indx]

    plt.plot(freq,suavizada,label = 'Ventana {}'.format(n+1))

    plt.vlines(x = freq[np.argmax(suavizada)], ymin = 0, ymax = max(suavizada),
           colors = 'green' if n < 10 else 'red',
           label = 'P.{} : {}Hz'.format(n+1,int(freq[np.argmax(suavizada)])))

    return freq, suavizada


def extract_features(freq, suavizada):
    amplitude_rms = np.sqrt(np.mean(np.square(suavizada)))

    freq_media = np.mean(freq * suavizada)

    freq_mediana = np.median(freq * suavizada)
    
    features = {
        'amplitude_rms': amplitude_rms,
        'freq_media': freq_media,
        'freq_mediana': freq_mediana
    }

    return features

if __name__ == "__main__":
    # Pruebas de ramas
    # Parámetros/variables globales
    Fs = 200
    tT = 120
    N_ventanas = 15
    dV = (Fs*tT)/N_ventanas

    N = Fs*tT
    n = np.arange(N)
    T = N/Fs
    freq = (n/T)

    frecN = int((Fs*T)/2)

    emgs, time = loadEMGs("selected_emgs/")
    df = pd.DataFrame()
    plt.figure(figsize=(15,12))

    segmentos_totales = []
    frecuencias_totales = []
    for m in range(10):
        segmentos = []
        frecuencias = []
        for n in range(N_ventanas):
            freq, suavizada = plot_emg_window(emgs[m],n,dV,smooth_window=500,int_freqs=True)

            features = extract_features(freq, suavizada)

            data = {
            'señal_suavizada': suavizada,
            'frecuencia': freq[np.argmax(suavizada)],
            'amplitude_rms': features['amplitude_rms'],
            'freq_media': features['freq_media'],
            'freq_mediana': features['freq_mediana']
        }
            df = df._append(data, ignore_index=True)
            frecuencias.append(freq[np.argmax(suavizada)])
            segmentos.append(suavizada)
        segmentos_totales.append(segmentos)
        frecuencias_totales.append(frecuencias)
    
    etiquetas = []

    # crieterio para etiquetado, básicamente el decremento en frecuencia
    for value in frecuencias_totales:
        for j in range(len(value)):
            etiqueta = 0 if value[0] - value[j] > 8 else 1
            etiquetas.append(etiqueta)

    print(len(etiquetas))
    df["etiquetas"] = etiquetas
    print(df.head(16))
    
    df.to_csv("data/processed_data.csv", index=False)
    
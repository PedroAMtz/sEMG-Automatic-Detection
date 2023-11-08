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

def process_signal(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Parámetros de la señal
    Fs = 200
    T = 120
    ventanas = 22
    windowsize = 150

    # Cálculos iniciales
    dV = (Fs * T) / ventanas
    N = Fs * T
    n = np.arange(N)
    T = N / Fs
    freq = n / T
    frecN = int((Fs * T) / 2)

    # Transformada rápida de Fourier
    X = fft(signal)
    emg_fft_abs = np.abs(X)
    emg_fft_abs[0] = 0

    # Envolvente superior
    df = pd.DataFrame(data={"y": emg_fft_abs})
    df["y_upperEnv"] = df["y"].rolling(window=windowsize).max().shift(int(-windowsize / 2))
    envolvente_superior = df['y_upperEnv'].to_numpy()

    return envolvente_superior, emg_fft_abs, freq

def smooth_signal(envolvente: np.ndarray):

    smoother_signal = savgol_filter(envolvente, 1001, 5)
    return smoother_signal

def plot_emg_window(emg_,n,dV,smooth_window=150,order=2,int_freqs=False):
    p_inicio = int(n*dV)
    p_final  = int((n+1)*dV)

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

    mask_0_50 = (freq >= 0) & (freq < 50)
    mask_50_100 = (freq >= 50) & (freq < 100)
    potencia_0_50 = np.sum(suavizada[mask_0_50])
    potencia_50_100 = np.sum(suavizada[mask_50_100])
    ratio_potencia = potencia_0_50 / potencia_50_100
    
    features = {
        'amplitude_rms': amplitude_rms,
        'freq_media': freq_media,
        'freq_mediana': freq_mediana,
        'ratio_potencia': ratio_potencia
    }

    return features

if __name__ == "__main__":
    # Pruebas de ramas
    # Parámetros/variables globales
    Fs = 200
    tT = 120
    N_ventanas = 22
    dV = (Fs*tT)/N_ventanas

    N = Fs*tT
    n = np.arange(N)
    T = N/Fs
    freq = (n/T)

    frecN = int((Fs*T)/2)

    emgs, time = loadEMGs("selected_emgs/")

    plt.figure(figsize=(15,12))

    for m in range(10):
        for n in range(N_ventanas):
            if n > 0 and n < N_ventanas-1:
                continue
            plt.subplot(5,2,m+1)
            freq, suavizada = plot_emg_window(emgs[m],n,dV,smooth_window=500,int_freqs=True)
            features = extract_features(freq, suavizada)
            print(features)

        plt.title('EMG {}'.format(m+1),fontsize=10)
        plt.legend()
    plt.show()
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


if __name__ == "__main__":
    emgs, time = loadEMGs("selected_emgs/")
    envolvente_sup, emg_fft, frequency = process_signal(emgs[0])
    new_signal = smooth_signal(envolvente_sup)   

    plt.figure(figsize=(10,5))

    plt.plot(frequency,emg_fft)
    plt.plot(frequency,new_signal)
    plt.xlim(5,100)
    plt.ylim(0,5000)
    plt.show()
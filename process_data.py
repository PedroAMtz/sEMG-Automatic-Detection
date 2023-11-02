import numpy as np
import os
from typing import Tuple


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

if __name__ == "__main__":
    emgs, time = loadEMGs("selected_emgs/")
    print(emgs.shape)
    print(time.shape)
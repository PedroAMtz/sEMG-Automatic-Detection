import pandas as pd
import argparse
import numpy as np
from process_data import loadEMGs, extract_features, plot_emg_window
import lightgbm as lgb
import os

def process_signal(signal: str) -> pd.DataFrame:
    Fs = 200
    tT = 120
    N_ventanas = 15
    dV = (Fs*tT)/N_ventanas

    N = Fs*tT
    n = np.arange(N)
    T = N/Fs
    freq = (n/T)
    df = pd.DataFrame()

    for n in range(N_ventanas):
        freq, suavizada = plot_emg_window(signal,n,dV,smooth_window=500,int_freqs=True)

        features = extract_features(freq, suavizada)

        data = {
            'señal_suavizada': suavizada,
            'frecuencia': freq[np.argmax(suavizada)],
            'amplitude_rms': features['amplitude_rms'],
            'freq_media': features['freq_media'],
            'freq_mediana': features['freq_mediana']
        }
        df = df._append(data, ignore_index=True)
    return df

def infer_fatigue(data: pd.DataFrame):
    X = data.drop(columns=["señal_suavizada"]).values
    model = lgb.Booster(model_file="models/fatigue_detection_v1.txt")
    predictions = model.predict(X)

    prediction_classes = []
    for prediction in predictions:
        prediction_class = np.argmax(prediction)
        prediction_classes.append(prediction_class)
    return prediction_classes

def list_emg_files(folder_path):
    emg_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    return emg_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform fatigue inference on a user-provided EMG signal.')
    parser.add_argument('input_file', type=str, help='Path to the input EMG signal file')

    args = parser.parse_args()
    print("------------- Initializing inference procedure -----------------")
    emg_files = list_emg_files(args.input_file)

    print("Available EMG files:")
    for i, emg_file in enumerate(emg_files, start=1):
        print(f"{i}. {emg_file}")

    user_choice = int(input("Enter the number corresponding to the EMG file you want to evaluate: ")) - 1

    if 0 <= user_choice < len(emg_files):
        emgs, time = loadEMGs(args.input_file)

        signal = emgs[user_choice]

        features = process_signal(signal)
        preds = infer_fatigue(features)

        if preds:
            print(f"\n Total number of windows applied to the input signal: {len(preds)} \n")
            print(f"Signal segments where fatigue transition was detected: {np.where(np.array(preds) == 0)[0]}")
            print(f"Window/s with fatigue: {np.where(np.array(preds) == 0)[0] + 1}")
    else:
        print("Invalid choice. Please enter a valid number.")

    
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve


with open(f'data/test_data.npy', 'rb') as f:
    X = np.load(f, allow_pickle=True)
    y = np.load(f, allow_pickle=True)

# load trained model
model = lgb.Booster(model_file="models/fatigue_detection_v1.txt")

if __name__ == "__main__":

    predictions = model.predict(X)

    prediction_classes = []
    for prediction in predictions:
        prediction_class = np.argmax(prediction)
        prediction_classes.append(prediction_class)
    
    y_pred_proba = predictions[::,1]

    fpr, tpr, thresholds = roc_curve(y,  y_pred_proba)
    auc = roc_auc_score(y, y_pred_proba)
    print(f"AUC: {auc} \n")
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'y--')
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('FP Rate')
    plt.ylabel('TP Rate')
    plt.title('ROC Curve')
    plt.show()
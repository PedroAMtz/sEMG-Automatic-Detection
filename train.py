import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb

data = pd.read_csv("data/processed_data.csv")
data = data.drop(columns=["señal_suavizada"])
train_data, test_data = train_test_split(data, test_size=0.2, random_state=34)

target_name = "etiquetas"

y_train = train_data[target_name].values
x_train = train_data.drop(columns=[target_name]).values

y_test = test_data[target_name].values
x_test = test_data.drop(columns=[target_name]).values

features = data.drop(columns=["etiquetas"]).columns.tolist()

if __name__ == "__main__":
    
    lgb_train = lgb.Dataset(x_train, y_train, feature_name = features)
    lgb_test = lgb.Dataset(x_test, y_test, feature_name = features)

    params = {
    'task': 'train'
    , 'boosting_type': 'gbdt'
    , 'objective': 'multiclass'
    , 'num_class': 2
    , 'metric': 'multi_logloss'
    }

    #gbm = lgb.train(params, lgb_train, num_boost_round=150, valid_sets=[lgb_test])
    #gbm.save_model("models/fatigue_detection_v1.txt")

    with open(f'data/test_data.npy', 'wb') as f:
        np.save(f, x_test)
        np.save(f, y_test)



#!usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
from xgboost.sklearn import XGBClassifier as XGBC
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def main():
    # パイプライン構築
    pipe = Pipeline([('preprocessing', StandardScaler()),
                     ('feature_selection', SelectFromModel(
                         RFC(n_estimators=20), threshold="median")),
                     ("pca", PCA(n_components=0.8)), ('classifier', XGBC(nthread=32))])

    # パラメータの設定
    param_grid = [
        {
            'classifier__max_depth': [5],
            'classifier__min_child_weight': [10],
            'classifier__gamma': [0.4],
            'classifier__subsample': [0.9],
            'classifier__colsample_bytree': [1.0],
            'classifier__reg_alpha': [0.07],
            'classifier__eta': [0.8],
            'classifier__lambda': [1],
            'classifier__alpha': [1.5],
            'classifier__objective': ['binary:logistic'],
            'classifier__eval_metric': ["rmse"]
        }
    ]

    gclf = GridSearchCV(pipe, param_grid, cv=5, n_jobs=32,
                        verbose=False, scoring='roc_auc')
    # データのロード
    X_train = np.loadtxt("train_data.csv", delimiter=",", skiprows=1,
                         usecols=[i for i in range(1, 24)])
    X_test = np.loadtxt("test_data.csv", delimiter=",", skiprows=1,
                        usecols=[i for i in range(1, 24)])
    y_train = np.loadtxt("train_data.csv", delimiter=",", skiprows=1,
                         usecols=(24))

    # GridSearchCVでハイパーパラメータを決定
    # KFoldでAccuracyを検証
    first_fold = True
    acc_ave = 0
    epoch = 0
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X_train, y_train):
        if first_fold:
            gclf.fit(X_train[train_index], y_train[train_index])
            clf = gclf.best_estimator_
            first_fold = False
            print(gclf.best_params_)
        clf.fit(X_train[train_index, ], y_train[train_index])

        acc = clf.score(X_train[test_index], y_train[test_index])
        acc_ave = acc_ave + acc
        epoch = epoch + 1

    print('Accuracy: {}'.format(acc_ave/epoch))

    # 訓練用データの正確性
    y_pred = clf.predict(X_train)
    cm = confusion_matrix(y_train, y_pred)
    print(cm)

    # テストデータで予測
    pred = clf.predict(X_test)

    # csv形式に加工
    submit = pd.DataFrame(
        {"ID": [i for i in range(0, 3000)], "Y": pred.astype(int)})
    submit.to_csv("submit.csv", index=None)


if __name__ == "__main__":
    main()

# 参考文献
# https://www.haya-programming.com/entry/2018/02/22/234011#%E4%BA%A4%E5%B7%AE%E6%A4%9C%E8%A8%BC
# http://datanerd.hateblo.jp/entry/2017/09/15/160742

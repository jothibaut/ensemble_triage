'''
Within this script, we validate the parameters of different ensemble algorithms.
'''

from sklearn.metrics import *

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import os
import pandas as pd
import time
import pickle

from machine_learning import scale_data
import whole_period
from commons import *


SEED = 42
MODEL_PARAMS = {
    'hist_gradient_boosting': {
        'model': HistGradientBoostingClassifier(random_state=SEED),
        'params': {'learning_rate': [0.1, 0.2],
                  'max_iter': [100, 200],
                  'max_leaf_nodes': [3, 7, 15, 31],
                  'min_samples_leaf': [15, 20, 25],
                  'tol': [1e-8, 1e-7]}
    },
    'gradient_boosting': {
        'model': GradientBoostingClassifier(random_state=SEED),
        'params': {'learning_rate': [0.05, 0.1, 0.2],
                   'n_estimators': [50, 75, 100],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [15, 20, 25],
                   'max_depth': [2, 3, 5]}
    },
    'xgboost': {
        'model': XGBClassifier(random_state=SEED),
        'params': {'validate_parameters': [False],
                   'learning_rate': [0.2, 0.3, 0.4],
                   'gamma': [0, 0.1, 5, 10],
                   'max_depth': [3, 6, 9],
                   'sampling_method': ['uniform', 'gradient_based'],
                   'tree_method': ['auto', 'exact', 'approx', 'hist'],
                   'max_bin': [128, 256]}
    },
    'light_gbm': {
        'model': LGBMClassifier(random_state=SEED),
        'params': {'boosting_type': ['gbdt', 'dart'],
                   'num_leaves': [7, 15, 31],
                   'max_depth': [3, 6, 9],
                   'learning_rate': [0.05, 0.1, 0.2],
                   'n_estimators': [50, 75, 100],
                   'subsample_for_bin':[32, 64, 200000]}
    },
    'cat_boost': {
        'model': CatBoostClassifier(logging_level='Silent', random_state=SEED),
        'params': {'iterations': [50, 75, 100],
                   'learning_rate': [0.2, 0.3, 0.4],
                   'depth': [3, 6, 9]}
    },
    'adaboost': {
        'model': AdaBoostClassifier(random_state=SEED),
        'params': {'n_estimators': [25, 50, 75, 100, 150, 200],
                   'learning_rate': [0.1, 0.5, 1.0, 2.0, 10.0],
                   'algorithm': ['SAMME', 'SAMME.R']}
    },
    'random_forest': {
        'model': RandomForestClassifier(random_state=SEED),
        'params': {'n_estimators': [50, 75, 100, 150, 200],
                   'max_depth': [None, 3, 5, 7],
                   'criterion': ['gini', 'log_loss'],
                   'min_samples_split': [2, 3, 5],
                   'min_samples_leaf': [1, 2, 5],
                   'bootstrap': [True, False],
                   'max_samples':[None, 0.4, 0.6, 0.8],
                   'max_features': ["sqrt", "log2"]}
    },
    'extra_trees': {
        'model': ExtraTreesClassifier(random_state=SEED),
        'params': {'n_estimators': [50, 75, 100, 150, 200],
                   'max_depth': [None, 3, 5, 7],
                   'criterion': ['gini', 'log_loss'],
                   'min_samples_split': [2, 3, 5],
                   'min_samples_leaf': [1, 2, 5],
                   'bootstrap': [True, False],
                   'max_samples':[None, 0.4, 0.6, 0.8],
                   'max_features': ["sqrt", "log2"]}
    }
}

'''
This function preprocesses the data set to make it in the format for training the ML model.
It is a simplification of the preprocessing procedure implemented within preprocessing.split_by_period().
@preprocessed_path: Input df.
@period: The period during we keep the data.
@imputed_path: Output of imputed df.
@scaled_path: Output of imputed and scaled df.
'''
def simplififed_preprocess_for_ncv(preprocessed_path, period, imputed_path, scaled_path):
    # NB: Mixed types were checked. This sometimes happen because of NA values.
    df = pd.read_csv(preprocessed_path)

    if 'test_criterion' in df.columns:
        df = df[df['test_criterion'] != 'positive_self_test']

    is_within_period = (df['test_date'] >= period[0]) & (df['test_date'] <= period[1])
    df_window = df[is_within_period]

    if period == WHOLE_PERIOD:
        assert df.shape[0] == df_window.shape[0], f"We missed some data while studying the whole period: {df.shape[0]} --> {df_window.shape[0]} obs"

    n_pos = df_window[df_window['result'] == True].shape[0]
    n = df_window.shape[0]
    print(f'Positive rate = {n_pos}/{n} = {n_pos / n:.2f}')

    df_imputed = whole_period.impute(df_window)

    df_pp = whole_period.preprocess(df_window)
    df_pp = scale_data(df_pp)

    df_imputed.to_csv(imputed_path, index=False)
    df_pp.to_csv(scaled_path, index=False)


'''
This function trains an ensemble model with a Nested Cross-Validation technique.
'''
def nested_cv(base_model, param_grid, X_train_val, X_test, y_train_val, y_test, feature_names, label, model_name):
    start_time = time.time()

    print('\nTRAINING MODEL : {0:s} ...'.format(model_name))

    # 3. Définir la validation croisée imbriquée
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    # Stocker les performances
    ap_scores = []
    auc_scores = []
    best_params_list = []

    # 6. Validation croisée imbriquée
    print("Starting Nested Cross-Validation...")
    for train_idx, val_idx in outer_cv.split(X_train_val, y_train_val):
        # Split en données train/val pour la boucle extérieure
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

        # Grid search dans la boucle intérieure
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            scoring="average_precision",
            cv=inner_cv,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Meilleurs hyperparamètres de la boucle intérieure
        best_params = grid_search.best_params_
        best_params_list.append(best_params)

        # Évaluer sur le test set extérieur
        best_model = grid_search.best_estimator_ # Trained on all 3/3 inner folds, NOT 2/3 inner folds.
        y_val_pred = best_model.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, y_val_pred)
        auc = roc_auc_score(y_val, y_val_pred)
        ap_scores.append(ap)
        auc_scores.append(auc)
        # print(f"Fold AP: {ap:.4f}")

    # Résultats globaux
    mean_ap = np.mean(ap_scores)
    std_ap = np.std(ap_scores)
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    # 7. Choisir les meilleurs hyperparamètres finaux
    print("Selecting final hyperparameters...")
    final_grid_search = GridSearchCV(
        base_model,
        param_grid,
        scoring="average_precision",
        cv=inner_cv,
        n_jobs=-1
    )
    final_grid_search.fit(X_train_val, y_train_val)

    # Meilleurs hyperparamètres finaux
    final_best_params = final_grid_search.best_params_

    # 8. Entraîner le modèle final
    final_model = final_grid_search.best_estimator_
    final_model.fit(X_train_val, y_train_val)

    # 9. Évaluer sur le test set
    y_test_pred = final_model.predict_proba(X_test)[:, 1]
    final_test_ap = average_precision_score(y_test, y_test_pred)
    final_test_auc = roc_auc_score(y_test, y_test_pred)

    # 10. Save y_test and y_test_pred into a DataFrame
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_test_pred
    })

    results_df.to_csv(os.path.join(DATA, SECONDARY, f'{label}_{model_name}_predictions.csv'), index=False)

    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_test_df.insert(loc=0, column='result', value=y_test)
    X_test_df.insert(loc=0, column='result_pred', value=y_test_pred)
    with open(os.path.join(DATA, SECONDARY, f"{label}_{model_name}_X_test.pkl"), "wb") as file:
        pickle.dump(X_test_df, file)

    with open(os.path.join(DATA, TRAINED_MODELS, f"{label}_{model_name}_model.pkl"), "wb") as file:
        pickle.dump(final_model, file)

    end_time = time.time()
    execution_time = end_time - start_time

    with open(os.path.join(DATA, ENSEMBLE, f"{label}_{model_name}_ncv_outputs.txt"), "w") as file:
        print(f'### {model_name} - NCV evaluations ###', file=file)
        print(f"Outer validation: Mean AP = {mean_ap:.4f}, Std ROC AUC = {std_ap:.4f}", file=file)
        print(f"Outer validation: Mean ROC AUC = {mean_auc:.4f}, Std ROC AUC = {std_auc:.4f}", file=file)
        print(f"Final Test AP: {final_test_ap:.4f}", file=file)
        print(f"Final Test ROC AUC: {final_test_auc:.4f}", file=file)
        print(f"Final parameters: {final_best_params}", file=file)
        print(f"{model_name} - Execution time: {execution_time/60:.2f} minutes", file=file)


'''
If you only want to validate the parameters of one model in the list, specify it in @the_one_model.
'''
def execute_nested_cv(path, label, the_one_model=None):
    X, y = load_dataset(path)
    # X = scale_data(X)

    feature_names = X.columns

    X = X.values
    y = y.values

    # 2. Diviser en train/val et test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    for model_name, mp in MODEL_PARAMS.items():
        if the_one_model is None:
            nested_cv(mp['model'], mp['params'], X_train_val, X_test, y_train_val, y_test, feature_names, label,
                      model_name)
        else:
            if the_one_model == model_name:
                nested_cv(mp['model'], mp['params'], X_train_val, X_test, y_train_val, y_test, feature_names, label,
                          model_name)


def display_metrics(predictions_path):
    threshold = 0.5
    df = pd.read_csv(predictions_path)

    y_test = df['actual'].values
    y_pred = df['predicted'].values
    y_pred_binary = (y_pred >= threshold).astype(int)

    ap = average_precision_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred_binary)
    rec = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    print(f'Average Precision : {ap:.3f}')
    print(f'ROC AUC : {auc:.3f}')
    print(f'Precision : {prec:.3f}')
    print(f'Recall : {rec:.3f}')
    print(f'F1-score : {f1:.3f}')


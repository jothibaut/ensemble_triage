'''
This analysis studies the evolultion over time of:
- The machine learning performance.
- The epidemic impact of triage.
- The influence of individual questions.
'''

from commons import *
from shared import  *

from preprocessing import impute_missing_data
from machine_learning import scale_data
from impact_tests import metrics_vectors
from impact_epi import compute_re


import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import shap
import ast

from xgboost import XGBClassifier
from sklearn.metrics import *

import joblib

FPR_I = 0.03
FNR_R = 0.1


'''
This function computes the number of tests performed at the test centre weekly.
'''
def weekly_test_number():
    df = pd.read_csv(os.path.join(DATA, RAW, 'all_tests.csv')) #TODO: Check this source, taken from dicotra/5-analysis/data/all_tests.csv. Probably comes from the line before, in comments in the code.

    df['test_date'] = pd.to_datetime(df['test_date'])
    df['week_start'] = df['test_date'].dt.to_period('W').apply(lambda r: r.start_time)

    weekly_counts = (
        df.groupby('week_start')['result']
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    weekly_counts.to_csv(os.path.join(DATA, SECONDARY, 'weekly_test_numbers.csv'), index=False)


'''
This function computes the number of tests performed at the test centre weekly.
'''
def weekly_positive_tests():
    df = pd.read_csv(os.path.join(DATA, RAW, 'all_tests.csv')) #TODO: Check this source, taken from dicotra/5-analysis/data/all_tests.csv. Probably comes from the line before, in comments in the code.

    df = df[df['result'] == 'positive']

    df['test_date'] = pd.to_datetime(df['test_date'])
    df['week_start'] = df['test_date'].dt.to_period('W').apply(lambda r: r.start_time)

    weekly_counts = (
        df.groupby('week_start')['result']
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    weekly_counts.to_csv(os.path.join(DATA, SECONDARY, 'weekly_positive_tests.csv'), index=False)


def impute(df):
    df = df.copy()

    for col in ['person_id', 'test_date', 'week_start']:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.dropna(axis=1, how='all')

    columns_ordered = df.columns
    df = impute_missing_data(df, 'mean')
    df = df[columns_ordered]

    return df


def add_average_re(result_df):
    df_re = pd.read_csv(os.path.join(DATA, IMPACT_EPI, 'BEL-re_estimates.csv'))
    df_re = df_re.loc[(df_re['data_type'] == 'Confirmed cases') & (df_re['estimate_type'] == 'Cori_slidingWindow'), ['date', 'median_R_mean']]

    df_re['date'] = pd.to_datetime(df_re['date'])
    df_re['week_start'] = df_re['date'].dt.to_period('W').apply(lambda r: r.start_time)

    weekly_avg_df = df_re.groupby('week_start', as_index=False)['median_R_mean'].mean()

    result_df = pd.merge(result_df, weekly_avg_df, how='left')

    return result_df


def add_patients_number(result_df, df):
    counts_df = df.groupby('week_start', as_index=False).size()
    counts_df.columns = ['week_start', 'n_patients']

    result_df = pd.merge(result_df, counts_df, how='left')

    return result_df


'''
--- Incubation period values ---
Ancestral strain: 6.3
2021-02-15 - Alpha: 5.0 (Grant et al. France)
2021-06-28 - Delta: 4.8 (pooled estimates)
2021-12-27 - Omicron: 3.6 (pooled estimates)
'''
def weekly_delays(window_size=4, window_shift=1):
    df = pd.read_csv(os.path.join(DATA, MERGED, '11-with_environment.csv'))[['person_id', 'test_date', 'result', 'symptoms_start_date']]

    df = df.dropna(subset=['symptoms_start_date'])
    df['symptoms_start_date'] = pd.to_datetime(df['symptoms_start_date'])

    prereg = pd.read_csv(os.path.join(DATA, RAW, 'preregistration.csv'))[['person_id', 'test_date', 'prereg_submit_time', 'test_time']]

    prereg['prereg_submit_time'] = pd.to_datetime(prereg['prereg_submit_time'])
    prereg['test_time'] = pd.to_datetime(prereg['test_time'])

    labres = pd.read_csv(os.path.join(DATA, RAW, 'godata_labres.csv'))[['person_id', 'test_date', 'test_result_avail_time']]
    labres['test_result_avail_time'] = pd.to_datetime(labres['test_result_avail_time'])

    df = pd.merge(df, prereg, how='left')
    df = pd.merge(df, labres, how='left')

    df = df.sort_values(by=['test_date'], ascending=True)

    df = df.dropna(subset=['prereg_submit_time'])
    df = df.dropna(subset=['test_result_avail_time'])

    df['test_date'] = pd.to_datetime(df['test_date'])
    df['week_start'] = df['test_date'].dt.to_period('W').apply(lambda r: r.start_time)

    min_date = df['week_start'].min()
    max_date = df['week_start'].max()

    windows = []

    current = min_date
    while current + pd.Timedelta(weeks=window_size-1) <= max_date:
        start_wk1 = current
        wk_last = current + pd.Timedelta(weeks=window_size-1)
        wk3_fourth = start_wk1 + pd.Timedelta(weeks=int(3 * window_size / 4))
        label = wk_last.strftime('%Y-%m-%d')
        the_df = df[(df['week_start'] >= start_wk1) & (df['week_start'] <= wk_last)].copy()

        the_df['delay_symptoms_test'] = (the_df['test_time'] - the_df['symptoms_start_date']).dt.total_seconds() / (3600 * 24)  # in days

        is_outlier = (the_df['delay_symptoms_test'] < 0) | (the_df['delay_symptoms_test'] > 14)
        the_df = the_df[~is_outlier]

        # Compute delays in hours or days
        the_df['delay_symptoms_prereg'] = (the_df['prereg_submit_time'] - the_df['symptoms_start_date']).dt.total_seconds() / (3600 * 24) # in days
        the_df['delay_prereg_test'] = (the_df['test_time'] - the_df['prereg_submit_time']).dt.total_seconds() / (3600 * 24)  # in days
        the_df['delay_test_reception'] = (the_df['test_result_avail_time'] - the_df['test_time']).dt.total_seconds() / (3600 * 24)  # in days

        is_outlier = (the_df['delay_prereg_test'] < 0) | (the_df['delay_prereg_test'] > 14)
        the_df = the_df[~is_outlier]

        is_outlier = (the_df['delay_symptoms_prereg'] < -14) | (the_df['delay_symptoms_prereg'] > 14)
        the_df = the_df[~is_outlier]

        is_outlier = (the_df['delay_test_reception'] < 0) | (the_df['delay_test_reception'] > 14)
        the_df = the_df[~is_outlier]

        # Compute mean delays
        mean_delay_symptoms_prereg = the_df['delay_symptoms_prereg'].mean()
        mean_delay_prereg_test = the_df['delay_prereg_test'].mean()
        mean_delay_test_reception = the_df['delay_test_reception'].mean()

        # Incubation period
        # IP values: https://doi.org/10.1093/jtm/taac052
        # Dominant VOC periods: https://doi.org/10.3390/v14102301 + Genomic surveillance report.
        if label >= '2021-12-27': #Omicron
            incubation_time = 3.6
        elif label >= '2021-06-28': # Delta
            incubation_time = 4.8
        elif label >= '2021-02-15': # Alpha
            incubation_time = 5.0
        else: # Ancestral strain
            incubation_time = 6.3

        windows.append({
            'week_start': label,
            'delay_symptoms_prereg': mean_delay_symptoms_prereg,
            'delay_prereg_test': mean_delay_prereg_test,
            'delay_test_reception': mean_delay_test_reception,
            'incubation_time': incubation_time,
            'n_obs': the_df.shape[0]
        })

        current += pd.Timedelta(weeks=window_shift)

    # Convert to DataFrame
    delays_df = pd.DataFrame(windows)
    delays_df['week_start'] = pd.to_datetime(delays_df['week_start'])

    delays_df.to_csv(os.path.join(DATA, SECONDARY, 'delays.csv'), index=False)


def add_model_performance(result_df, df, prediction_folder, n_runs=2, window_size=4, window_shift=1):
    model = joblib.load(os.path.join(DATA, TRAINED_MODELS, 'whole_xgboost_model.pkl'))
    best_params = model.get_params()

    min_date = df['week_start'].min()
    max_date = df['week_start'].max()

    # Generate all possible windows with a 1-week shift
    windows = []
    current = min_date

    while current + pd.Timedelta(weeks=window_size-1) <= max_date:
        # Weeks: [wk1, wk2, wk3, wk4)
        start_wk1 = current
        wk_last = current + pd.Timedelta(weeks=window_size-1)
        wk3_fourth = start_wk1 + pd.Timedelta(weeks=int(3 * window_size / 4))
        label = wk3_fourth.strftime('%Y-%m-%d')

        df_window = df[(df['week_start'] >= start_wk1) & (df['week_start'] <= wk_last)]

        week_start_values = df_window['week_start'].values
        df_test_nopp = df_window[df_window['week_start'] >= wk3_fourth]
        df_test_nopp.to_csv(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, f'df_test_nopp_{label}.csv'), index=False)

        df_pp = preprocess(df_window)
        df_pp = scale_data(df_pp)
        df_pp.insert(loc=1, column='week_start', value=week_start_values)

        df_train = df_pp[df_pp['week_start'] < wk3_fourth]
        df_test = df_pp[df_pp['week_start'] >= wk3_fourth]

        df_train = df_train.drop('week_start', axis=1)
        df_test = df_test.drop('week_start', axis=1)

        scores = []
        for i in range(n_runs):
            X_train = df_train.iloc[:, 1:]
            X_test = df_test.iloc[:, 1:]
            y_train = df_train.iloc[:, 0]
            y_test = df_test.iloc[:, 0]

            if label == '2021-09-27':
                X_train.to_csv(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, f'X_train_{label}.csv'), index=False)
                y_train.to_csv(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, f'y_train_{label}.csv'), index=False)
                y_test.to_csv(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, f'y_test_{label}.csv'), index=False)

            pos_rate = y_test.mean()

            model = XGBClassifier(**best_params)
            model.set_params(gamma=0.1)

            model.fit(X_train, y_train)

            y_scores = model.predict_proba(X_test)[:, 1]

            predictions = pd.DataFrame({'actual': y_test, 'predicted': y_scores})
            predictions.to_csv(os.path.join(prediction_folder, f'predictions_{label}.csv'), index=False)
            with open(os.path.join(prediction_folder, f"X_test_{label}.pkl"), "wb") as file:
                pickle.dump(X_test, file)

            with open(os.path.join(prediction_folder, f"model_{label}.pkl"), "wb") as file:
                pickle.dump(model, file)

            ap_score = average_precision_score(y_test, y_scores)
            if y_test.nunique() == 2:
                auc_score = roc_auc_score(y_test, y_scores)
            else:
                auc_score = 0.0
            scores.append(ap_score) # Useful if we want to run the training several times at some point.

        windows.append({
            'week_start': label,
            'ap': ap_score,
            'roc_auc': auc_score,
            'n_patients': df_test.shape[0],
            'pos_rate': pos_rate
        })

        current += pd.Timedelta(weeks=window_shift)

    # Convert to DataFrame
    performance_df = pd.DataFrame(windows)
    performance_df['week_start'] = pd.to_datetime(performance_df['week_start'])

    result_df = pd.merge(performance_df, result_df, how='left')

    return result_df


def minimise_T(predictions_path, infected, healthy, FPR_max=None, FNR_max=None):
    assert (FPR_max is not None) and (FNR_max is not None), "Please provide values for both FPR_max and FNR_max."
    metrics = metrics_vectors(predictions_path)

    # Find isolation threshold for FPR_i <= FPR_max.
    i = 0
    while i+1 < metrics.shape[0] and metrics.loc[i+1, 'fpr'] <= FPR_max:
        i += 1

    # Find release threshold for FNR_r <= FNR_min.
    r = metrics.shape[0]-1
    while r-1 >= 0 and metrics.loc[r-1, '1-sens'] <= FNR_max:
        r -= 1

    # Compute the optimal number of tests, at the intersection of both lines.
    sens_i_T = metrics.loc[i, 'sensitivity']
    fpr_i_T = metrics.loc[i, 'fpr']
    sens_r_T = metrics.loc[r, 'sensitivity']
    fpr_r_T = metrics.loc[r, 'fpr']

    T_min = compute_n_tests(sens_i_T, sens_r_T, fpr_i_T , fpr_r_T, infected, healthy)

    return T_min


def triage(prediction_path, thr_release, thr_isolation):
    predictions = pd.read_csv(prediction_path)

    is_isolated = predictions["predicted"] >= thr_isolation
    is_released = (predictions["predicted"] < thr_release) & (predictions["predicted"] < thr_isolation)
    is_tested = ~is_isolated & ~is_released

    df_r = predictions[is_released]
    df_t = predictions[is_tested]
    df_i = predictions[is_isolated]

    n_r = df_r.shape[0]
    n_t = df_t.shape[0]
    n_i = df_i.shape[0]

    pos_t = df_t[df_t['actual']==True].shape[0]

    assert n_r + n_t + n_i == predictions.shape[0], "The number of patients sorted in triage categories does not match with the total number of patients."

    return n_r, n_t, n_i, pos_t


def add_epi_impact(result_df, FPR_max=0.03, FNR_min=None, T_max=500, FNR_max=0.1, stoch=False):
    dates = []
    Re = []
    Q1 = []
    Q2 = []
    T = []
    fnr = []
    release = []
    test = []
    isolate = []
    test_pos = []

    tests = pd.read_csv(os.path.join(DATA, SECONDARY, 'weekly_test_numbers.csv'))
    tests['week_start'] = pd.to_datetime(tests['week_start'])

    delays = pd.read_csv(os.path.join(DATA, SECONDARY, 'delays.csv'))
    delays['week_start'] = pd.to_datetime(delays['week_start'])

    result_df = pd.merge(result_df, tests, how='left')
    result_df = pd.merge(result_df, delays, how='left')
    # result_df = result_df[63:]

    for i in result_df.index:
        wk_start = result_df.loc[i, 'week_start']
        R_init = result_df.loc[i, 'median_R_mean']
        infected = result_df.loc[i, 'positive']
        healthy = result_df.loc[i, 'negative']
        sa = result_df.loc[i, 'delay_symptoms_prereg']
        at = result_df.loc[i, 'delay_prereg_test']
        tr = result_df.loc[i, 'delay_test_reception']
        ip = result_df.loc[i, 'incubation_time']

        label = wk_start.strftime('%Y-%m-%d')
        print(label)

        res = minimise_R('XXX',
                         os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, f'predictions_{label}.csv'),
                         R_init, sa, at, tr, ip,
                         infected, healthy,
                         FPR_max=FPR_max, FNR_min=FNR_min, T_max=T_max, stoch=stoch, n_inf=round(infected/7))

        if stoch: #Re_min is now the average avlues among all stochastic iterations.
            Re_ranges, fnr_r_R, thr_r_R, thr_i_R = res
        else:
            Re_min, fnr_r_R, thr_r_R, thr_i_R = res

        T_min = minimise_T(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, f'predictions_{label}.csv'), infected, healthy,
                           FPR_max=FPR_max, FNR_max=FNR_max)

        # We save the triage decision
        n_r, n_t, n_i, pos_t = triage(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, f'predictions_{label}.csv'), thr_r_R, thr_i_R)

        dates.append(wk_start)
        T.append(T_min)
        fnr.append(fnr_r_R)
        release.append(n_r)
        test.append(n_t)
        isolate.append(n_i)
        test_pos.append(pos_t)

        if stoch:
            Re.append(Re_ranges[0])
            Q1.append(Re_ranges[1])
            Q2.append(Re_ranges[2])
        else:
            Re.append(Re_min)

    if stoch:
        intermediate_df = pd.DataFrame({'week_start': dates,
                                        'Re_mean': Re,
                                        'Q1': Q1,
                                        'Q2': Q2,
                                        'T': T,
                                        'FNR_B': fnr,
                                        'release': release,
                                        'test': test,
                                        'isolate': isolate,
                                        'test_pos': test_pos})
    else:
        intermediate_df = pd.DataFrame({'week_start': dates,
                                        'Re': Re,
                                        'T': T,
                                        'FNR_B': fnr,
                                        'release': release,
                                        'test': test,
                                        'isolate': isolate,
                                        'test_pos': test_pos})

    result_df = pd.merge(result_df, intermediate_df, how='left')

    return result_df


def add_epi_impact_list(result_df, FPR_max_list=None, FNR_min_list=None, T_max_list=None, FNR_max_list=None, history_list=None, phases_list=None, stoch=False):
    assert len(history_list) == len(FPR_max_list)+1 == len(FNR_min_list)+1 == len(FNR_max_list)+1 == len(phases_list)+1, "The history of threshold is not clearly defined."

    dates = []
    Re = []
    Q1 = []
    Q2 = []
    T = []
    fnr = []
    release = []
    test = []
    isolate = []
    test_pos = []
    phase = []

    tests = pd.read_csv(os.path.join(DATA, SECONDARY, 'weekly_test_numbers.csv'))
    # tests['week_start'] = pd.to_datetime(tests['week_start'])

    delays = pd.read_csv(os.path.join(DATA, SECONDARY, 'delays.csv'))
    # delays['week_start'] = pd.to_datetime(delays['week_start'])

    result_df = pd.merge(result_df, tests, how='left')
    result_df = pd.merge(result_df, delays, how='left')

    result_df = result_df[(result_df['week_start']>=history_list[0]) & (result_df['week_start']<=history_list[-1])]
    result_df = result_df.reset_index(drop=True)

    i = 0
    j = 0
    wk_start = result_df.loc[0, 'week_start']
    while i < result_df.shape[0]:
        while wk_start < history_list[j+1]:
            wk_start = result_df.loc[i, 'week_start']
            R_init = result_df.loc[i, 'median_R_mean']
            infected = result_df.loc[i, 'positive']
            healthy = result_df.loc[i, 'negative']
            sa = result_df.loc[i, 'delay_symptoms_prereg']
            at = result_df.loc[i, 'delay_prereg_test']
            tr = result_df.loc[i, 'delay_test_reception']
            ip = result_df.loc[i, 'incubation_time']
            FPR_max = FPR_max_list[j]
            FNR_min = FNR_min_list[j]
            FNR_max = FNR_max_list[j]
            the_phase = phases_list[j]

            # label = wk_start.strftime('%Y-%m-%d')
            label = wk_start
            print(label)

            res = minimise_R('XXX',
                             os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, f'predictions_{label}.csv'),
                             R_init, sa, at, tr, ip,
                             infected, healthy,
                             FPR_max=FPR_max, FNR_min=FNR_min, T_max=None, stoch=stoch, n_inf=round(infected/6))

            if stoch: #Re_min is now the average avlues among all stochastic iterations.
                Re_ranges, fnr_r_R, thr_r_R, thr_i_R = res
            else:
                Re_min, fnr_r_R, thr_r_R, thr_i_R = res

            T_min = minimise_T(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, f'predictions_{label}.csv'), infected, healthy,
                               FPR_max=FPR_max, FNR_max=FNR_max)

            # We save the triage decision
            n_r, n_t, n_i, pos_t = triage(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, f'predictions_{label}.csv'), thr_r_R, thr_i_R)

            dates.append(wk_start)
            T.append(T_min)
            fnr.append(fnr_r_R)
            release.append(n_r)
            test.append(n_t)
            isolate.append(n_i)
            test_pos.append(pos_t)
            phase.append(the_phase)

            if stoch:
                Re.append(Re_ranges[0])
                Q1.append(Re_ranges[1])
                Q2.append(Re_ranges[2])
            else:
                Re.append(Re_min)
            i = i+1
        j = j+1

    if stoch:
        intermediate_df = pd.DataFrame({'week_start': dates,
                                        'phase': phase,
                                        'Re_mean': Re,
                                        'Q1': Q1,
                                        'Q2': Q2,
                                        'T': T,
                                        'FNR_B': fnr,
                                        'release': release,
                                        'test': test,
                                        'isolate': isolate,
                                        'test_pos': test_pos})
    else:
        intermediate_df = pd.DataFrame({'week_start': dates,
                                        'phase': phase,
                                        'Re': Re,
                                        'T': T,
                                        'FNR_B': fnr,
                                        'release': release,
                                        'test': test,
                                        'isolate': isolate,
                                        'test_pos': test_pos})

    result_df = pd.merge(result_df, intermediate_df, how='left')

    return result_df


def add_classic_epi_impact(result_df):
    dates = []
    Re = []

    tests = pd.read_csv(os.path.join(DATA, SECONDARY, 'weekly_test_numbers.csv'))
    tests['week_start'] = pd.to_datetime(tests['week_start'])

    delays = pd.read_csv(os.path.join(DATA, SECONDARY, 'delays.csv'))
    delays['week_start'] = pd.to_datetime(delays['week_start'])

    result_df = pd.merge(result_df, tests, how='left')
    result_df = pd.merge(result_df, delays, how='left')

    for i in result_df.index:
        wk_start = result_df.loc[i, 'week_start']
        R_init = result_df.loc[i, 'median_R_mean']

        sa = result_df.loc[i, 'delay_symptoms_prereg']
        at = result_df.loc[i, 'delay_prereg_test']
        tr = result_df.loc[i, 'delay_test_reception']
        ip = result_df.loc[i, 'incubation_time']

        re = compute_re("XXX", -1000, -1000, R_init, sa, at, tr, ip, "classic")

        dates.append(wk_start)
        Re.append(re)

    intermediate_df = pd.DataFrame({'week_start': dates,
                                    'Re_classic': Re})

    result_df = pd.merge(result_df, intermediate_df, how='left')

    return result_df


def add_relevant_questions(result_df):
    dates = []
    relevant_questions_lists = []

    for i in result_df.index:
        wk_start = result_df.loc[i, 'week_start']
        label = wk_start.strftime('%Y-%m-%d')

        with open(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, f"model_{label}.pkl"), "rb") as file:
            model = pickle.load(file)
        with open(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, f"X_test_{label}.pkl"), "rb") as file:
            X_test = pickle.load(file)

        X_test = X_test.astype(float)
        X_test_sample = X_test.sample(n=min(X_test.shape[0], 500), random_state=42)  # 500

        # Create the SHAP explainer for the Random Forest model
        explainer = shap.TreeExplainer(model)

        # Compute SHAP values for the test set
        shap_values = explainer.shap_values(X_test_sample)

        f, ax = plt.subplots(figsize=(8, 12))
        shap.summary_plot(shap_values, features=X_test_sample, max_display=10, show=False, plot_type='dot', color_bar=False)
        yticklabels = [tick.get_text() for tick in ax.get_yticklabels()]
        yticklabels = yticklabels[::-1]

        relevant_questions = []
        for label in yticklabels:
            matched = False
            for prefix in CATEGORICAL_COLUMNS:
                if label.startswith(prefix):
                    candidate = prefix
                    matched = True
                    break
            if not matched:
                candidate = label

            if candidate in OPTIONAL_FEATURES:
                relevant_questions.append(candidate)

        # If you want to remove duplicates after replacement
        relevant_questions = list(dict.fromkeys(relevant_questions))[:3]

        # my_shap_summary_plot(shap_values, X_test_sample, "tmp.png", show=False, max_display=10)

        dates.append(wk_start)
        relevant_questions_lists.append(relevant_questions)

    intermediate_df = pd.DataFrame({'week_start': dates,
                                    'relevant_questions': relevant_questions_lists})

    result_df = pd.merge(result_df, intermediate_df, how='left')

    return result_df


def epi_impact_evolution_data(date_col='test_date', exclude_positive_selftests=True):
    df = pd.read_csv(os.path.join(DATA, PREPROCESSED, 'preprocessed.csv'))

    if exclude_positive_selftests:
        df = df[df['test_criterion'] != 'positive_self_test']

    n_pos = df[df['result'] == True].shape[0]
    n = df.shape[0]
    print(f'Positive rate = {n_pos}/{n} = {n_pos/n:.3f}')

    is_within_period = (df['test_date'] >= WHOLE_PERIOD[0]) & (df['test_date'] <= WHOLE_PERIOD[1])
    # is_within_period = (df['test_date'] >= '2021-09-01') & (df['test_date'] <= '2021-10-07')
    df = df[is_within_period]

    df[date_col] = pd.to_datetime(df[date_col])
    df['week_start'] = df[date_col].dt.to_period('W').apply(lambda r: r.start_time)

    result_df = pd.DataFrame({'week_start': sorted(df['week_start'].unique())})

    result_df = add_average_re(result_df)
    result_df = add_model_performance(result_df, df, os.path.join(DATA, WHOLE_PERIOD_ANALYSIS), n_runs=1)
    print(result_df.columns)
    result_df['week_start'] = pd.to_datetime(result_df['week_start'])
    result_df['week_start'] = result_df['week_start'].dt.to_period('W').apply(lambda r: r.start_time)
    result_df = add_classic_epi_impact(result_df)
    print(result_df.columns)
    result_df = add_relevant_questions(result_df)
    print(result_df.columns)
    result_df.to_csv(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, 'data_ml.csv'), index=False)

    generate_intermediate_files()


def visualise_ml_impact_evolution(data_path, figure_path, title):
    df = pd.read_csv(data_path)

    df.loc[df['Re'] == 0, 'Re'] = np.nan
    df.loc[df['roc_auc'] == 0, 'roc_auc'] = np.nan

    fontsize = 20
    ticksize = 20

    col_line = INDIGO

    fig, ax1 = plt.subplots(figsize=(16, 6))

    # Primary axis: line plot for median_R_mean

    ax1.plot(df['week_start'], df['roc_auc'], marker='*', color=col_line, label='ROC AUC')
    ax1.set_xlabel('Week', fontsize=fontsize)
    ax1.set_ylabel('Machine learning\nperformance', fontsize=fontsize, color=col_line)
    ax1.tick_params(axis='y', labelsize=ticksize, colors=col_line)
    ax1.set_ylim(bottom=0)

    # X-axis ticks (every 4th week)
    xticks = df['week_start'][::4]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=ticksize)

    # Secondary axis: stacked bar plot for release, test, isolate
    ax2 = ax1.twinx()

    bottom = np.zeros(len(df))
    for category, color in zip(['release', 'test', 'isolate'],
                               # ['#4C72B0', '#55A868', '#C44E52']):  # Customize colors if needed
                               [TURQUOISE, INDIGO, SALMON]):  # Customize colors if needed
        ax2.bar(df['week_start'], df[category], bottom=bottom, label=category.capitalize(), color=color, alpha=0.4)
        bottom += df[category]

    ax2.set_ylabel('Number of students', fontsize=fontsize)
    ax2.set_ylim(0, 2200)  # (ymin, ymax)
    ax2.tick_params(axis='y', labelsize=ticksize)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=fontsize-4)

    plt.title(title, fontsize=fontsize)
    fig.tight_layout()
    plt.savefig(figure_path, dpi=300)


def visualise_relevant_questions_evolution(data_path, figure_path):
    fontsize = 18
    ticksize = 16
    legend_fontsize = 13

    df = pd.read_csv(data_path)
    df['relevant_questions'] = df['relevant_questions'].apply(ast.literal_eval)

    df['relevant_questions'] = df['relevant_questions'].apply(
        lambda lst: [LABEL_MAP.get(x, x) for x in lst]
    )

    # Gather all unique questions to assign consistent colors
    all_questions = sorted(set(q for q_list in df['relevant_questions'] for q in q_list))
    color_map = plt.get_cmap('tab20')
    color_dict = {q: color_map(i % 20) for i, q in enumerate(all_questions)}

    fig, ax = plt.subplots(figsize=(14, 6))

    # Scatter each question at y = 0, -1, -2 (top to bottom)
    for _, row in df.iterrows():
        week = row['week_start']
        for i, question in enumerate(row['relevant_questions']):
            ax.scatter(week, -i, color=color_dict[question], marker='s', label=question, s=37)

    # Legend: avoid duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Variables', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=legend_fontsize, title_fontsize=legend_fontsize)

    ax.set_yticks([0, -1, -2])
    ax.set_yticklabels(['1st', '2nd', '3rd'], fontsize=fontsize)
    ax.set_xlabel('Week Start', fontsize=fontsize)
    ax.set_title('Top 3 most influential variables over time', fontsize=fontsize)

    xticks = df['week_start'][::4]
    ax.set_xticks(xticks)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=ticksize)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)


def epi_impact_evolution(exclude_positive_selftests=True):
    epi_impact_evolution_data(exclude_positive_selftests=exclude_positive_selftests)
    visualise_relevant_questions_evolution(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, 'data_impact.csv'),
                                           os.path.join(FIGURES, WHOLE_PERIOD_ANALYSIS, 'questions_evolution.svg'))


def generate_intermediate_files():
    # Weekly metrics file
    df = pd.read_csv(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, 'data_ml.csv'))
    weeks = sorted(df["week_start"].unique())
    for week in weeks:
        print(week)
        predictions_path = os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, f"predictions_{week}.csv")
        metrics_path = os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, f"metrics_{week}.csv")
        metrics_vectors(predictions_path, metrics_path)


if __name__ == '__main__':
    # weekly_test_number()
    # weekly_positive_tests()
    # weekly_delays(window_size=4)
    # epi_impact_evolution(exclude_positive_selftests=True)

    epi_impact_evolution_data()
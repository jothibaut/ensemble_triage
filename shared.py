'''
This scripts contains various functions, useful for diverse analyses.
'''

from preprocessing import impute_missing_data

import math
import pandas as pd
import subprocess
from sklearn.metrics import *


CATEGORICAL_COLUMNS = ['test_criterion', 'vaccine_type', 'vaccination_status', 'travel_area', 'stopover_area',
                       'known_cases_area']
BOOLEAN_COLUMNS = ['sex', 'do_have_symptoms', 'source_event_known', 'source_person_known',  'other_unknown_attendants',
                   'student_residence', 'traveller', 'stopover', 'other_cases_in_residence', 'shared_kitchen',
                   'shared_bathroom', 'possible_superspreading', 'known_cases_in_area']


'''
This function preprocesses each time-windowed dataframe.
'''
def preprocess(df):
    df = df.copy()

    for col in ['person_id', 'test_date', 'week_start']:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.dropna(axis=1, how='all')

    columns_ordered = df.columns
    df = impute_missing_data(df, 'mean')
    df = df[columns_ordered]

    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    data_encoded = pd.get_dummies(df, columns=[col for col in CATEGORICAL_COLUMNS if col in columns_ordered])

    return data_encoded


def compute_n_tests(sens_i, sens_r, fpr_i, fpr_r, infected, healthy):
    n_infected = (sens_r - sens_i) * infected  # Number of tests for infected population.
    n_healthy = (fpr_r - fpr_i) * healthy  # Number of tests for healthy population.

    n = n_infected + n_healthy  # Total number of tests, as a proportion.

    if not math.isnan(n):
        n = int(n)

    return n


def compute_re(period_name, sens_i, sens_r, R_init, sa, at, tr, ip, strategy="ml", stoch=False, n_inf=20):
    if stoch and n_inf == 0:
        return 0, 0, 0

    if stoch:
        stoch_str = "TRUE"
    else:
        stoch_str = "FALSE"

    command = [
        "Rscript",
        "compute_Re.R",
        period_name,
        str(sens_i),
        str(sens_r),
        str(R_init),
        str(sa),
        str(at),
        str(tr),
        str(ip),
        strategy,
        stoch_str,
        str(n_inf)
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    # Get the output and convert to float
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if stdout != '':
        if stoch:
            try:
                mean_val_str, q1_str, q3_str = stdout.split(",")
                res = float(mean_val_str), float(q1_str), float(q3_str)
            except ValueError:
                print("Unexpected output from R:", stdout)
                res = 0, 0, 0
        else:
            res = float(stdout)
    else:
        print(stderr)
        if stoch:
            res = 0, 0, 0
        else:
            res = 0

    return res


def metrics_vectors(predictions_path, metrics_path=None):
    df = pd.read_csv(predictions_path)

    y_test = df['actual'].values
    y_test_pred = df['predicted'].values

    if y_test.sum() == 0:
        print('No positive samples')

    fpr, sens, thresholds = roc_curve(y_test, y_test_pred)

    assert len(thresholds) == len(sens) == len(fpr), f"Lengths of the [threshols, sensitivity, FPR] values do not match: [{len(thresholds)}, {len(sens)}, {len(fpr)}]"

    df_metrics = pd.DataFrame(data={'threshold': thresholds,
                                    'sensitivity': sens,
                                    'fpr': fpr,
                                    '1-fpr': 1-fpr,
                                    '1-sens': 1-sens}) #=FNR

    if metrics_path is not None:
        df_metrics.to_csv(metrics_path, index=False)

    return df_metrics


def minimise_R(period_name, predictions_path, R_init, sa, at, tr, ip, infected, healthy, FPR_max=None, FNR_min=None,
               T_max=None, stoch=False, n_inf=20):
    assert FPR_max is not None, "Please provide a FPR_max."
    assert (FNR_min is None) != (T_max is None), "Please provide one and only one values for variables FNR_min, T_max."

    metrics = metrics_vectors(predictions_path)

    # Find isolation threshold for FPR_i <= FPR_max
    i = 0
    while i+1 < metrics.shape[0] and metrics.loc[i+1, 'fpr'] < FPR_max:
        i += 1

    sens_i_R = metrics.loc[i, 'sensitivity']
    fpr_i_R = metrics.loc[i, 'fpr']
    thr_i_R = metrics.loc[i, 'threshold']

    # Find release threshold for FNR_r >= FNR_min
    if FNR_min is not None:
        if pd.isna(metrics.loc[0, '1-sens']): #FNR is nan in this case.
            r = metrics.shape[0] - 1
        else:
            r = 0

        while r+1 < metrics.shape[0] and metrics.loc[r+1, '1-sens'] >= FNR_min:
            r += 1
    # Find release threshold for T <= T_max
    elif T_max is not None:
        T = []
        for j in metrics.index:
            sens_r_j = metrics.loc[j, 'sensitivity']
            fpr_r_j = metrics.loc[j, 'fpr']
            n = compute_n_tests(sens_i_R, sens_r_j, fpr_i_R , fpr_r_j, infected, healthy)
            T.append(n)

        # Sometimes here, T is negative.
        # This is normal since we compute all possible values of n_tests on the line Sens_i = Sens_i_A.
        # Therefore, sometimes, the THR_isolate < THR_release, which is not realistic for our triage model.
        metrics['T'] = T

        mask = metrics['T'] < T_max

        # If there is no solution, assign r=-1
        if mask.any():
            r = metrics[mask]['T'].idxmax() # Find the intersection between the Tmax curve and the FPRmax line.
        else:
            r = -1

    if r == -1:
        if stoch:
            res = 0, 0, 0
        else:
            res = 0
        fnr_r_R = 0
        thr_r_R = 0
    else:
        sens_r_R = metrics.loc[r, 'sensitivity']
        fnr_r_R = metrics.loc[r, '1-sens']
        if pd.isna(fnr_r_R):
            thr_r_R = 1.0
        else:
            thr_r_R = metrics.loc[r, 'threshold']
        res = compute_re(period_name, sens_i_R, sens_r_R, R_init, sa, at, tr, ip, stoch=stoch, n_inf=n_inf)

    return res, fnr_r_R, thr_r_R, thr_i_R



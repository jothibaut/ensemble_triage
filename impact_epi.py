'''
Within this script, we study the epidemiological impact of implementing the machine learning model in real-life.
'''

from commons import *
from shared import *

from sklearn.metrics import *

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import time

import joblib
import pickle

from whole_period import preprocess
from machine_learning import scale_data
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


CONTROL_TRAIN_PERIOD_ALPHA = ('2021-02-01', '2021-03-11')
CONTROL_TEST_PERIOD_ALPHA = (CONTROL_TRAIN_PERIOD_ALPHA[1], '2021-03-25') # 2 weeks

CONTROL_TRAIN_PERIOD_DELTA = ('2021-04-05', '2021-09-27')
CONTROL_TEST_PERIOD_DELTA = (CONTROL_TRAIN_PERIOD_DELTA[1], '2021-10-11') # 2 weeks

CONTROL_TRAIN_PERIOD_BA1_EARLY = ('2021-09-20', '2021-12-15')
CONTROL_TEST_PERIOD_BA1_EARLY = (CONTROL_TRAIN_PERIOD_BA1_EARLY[1], '2022-02-07') # 8 weeks

CONTROL_TRAIN_PERIOD_BA1_LATE = ('2021-09-20', '2021-12-22')
CONTROL_TEST_PERIOD_BA1_LATE = (CONTROL_TRAIN_PERIOD_BA1_LATE[1], '2022-02-07') # 7 weeks

CONTROL_TRAIN_PERIOD_BA2 = ('2021-12-13', '2022-02-28')
CONTROL_TEST_PERIOD_BA2 = (CONTROL_TRAIN_PERIOD_BA2[1], '2022-04-04') # 5 weeks

training_periods = [CONTROL_TRAIN_PERIOD_ALPHA,
                    CONTROL_TRAIN_PERIOD_DELTA,
                    CONTROL_TRAIN_PERIOD_BA1_EARLY,
                    CONTROL_TRAIN_PERIOD_BA1_LATE,
                    CONTROL_TRAIN_PERIOD_BA2]
testing_periods = [CONTROL_TEST_PERIOD_ALPHA,
                   CONTROL_TEST_PERIOD_DELTA,
                   CONTROL_TEST_PERIOD_BA1_EARLY,
                   CONTROL_TEST_PERIOD_BA1_LATE,
                   CONTROL_TEST_PERIOD_BA2]
suffixes = ['_alpha',
            '_delta',
            '', # ba1_early is the main analysis. Its corresponding files were not labelled with a suffix.
            '_ba1_late',
            '_ba2']


'''
This function computes the average observed Re in Belgium, during on @the_date
@the_date: String
'''
def get_observed_re(the_date):
    df = pd.read_csv(os.path.join(DATA, IMPACT_EPI, 'BEL-re_estimates.csv'))

    df = df.loc[(df['data_type'] == 'Confirmed cases') & (df['estimate_type'] == 'Cori_slidingWindow'), ['date', 'median_R_mean']]

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    return df.loc[df['date'] == pd.to_datetime(the_date), 'median_R_mean'].values[0]


def sensitivity_vector(predictions_path, sensitivity_path):
    df = pd.read_csv(predictions_path)

    y_test = df['actual'].values
    y_test_pred = df['predicted'].values

    fpr, sens, thresholds = roc_curve(y_test, y_test_pred)

    df_s = pd.DataFrame(data={'threshold': thresholds,
                              'sensitivity': sens})

    df_s.to_csv(sensitivity_path, index=False)


def transform_matrix(re_matrix_path, new_re_matrix_path):
    df = pd.read_csv(re_matrix_path, index_col=0)
    df = df.T

    df.index = df.index.astype(float)
    df.index = 1 - df.index

    df.to_csv(new_re_matrix_path)


def visualize_re_matrix(re_matrix_path, figure_path):
    # Load the CSV file
    df = pd.read_csv(re_matrix_path, index_col=0)
    re_classic = 0.9427

    x_positions = np.array([])
    y_positions = np.array([])

    for i in range(df.shape[1]):
        # Find row where Re is closest to re_classic
        diff = (df.iloc[:, i] - re_classic).abs().values
        if not all(df.iloc[:, i].isna()):
            j = np.nanargmin(diff)
            if i != j+1 and j != 0:
                x_positions = np.append(x_positions, i)
                y_positions = np.append(y_positions, j)

    # Create heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(df, annot=False, cmap="coolwarm", cbar_kws={'label': 'Re Value'})

    # Adjust color bar font size
    cbar = ax.collections[0].colorbar
    cbar.set_label('Re Value', fontsize=16)


    x_positions = x_positions + 0.5
    y_positions = y_positions + 0.5
    ax.plot(x_positions, y_positions, color='black', linewidth=2, zorder=3, label=f"Re = {re_classic:.2f} with classic testing strategy")

    ax.invert_yaxis()
    ax.invert_xaxis()

    # Set tick labels
    ax.set_xticklabels([f"{1 - float(label.get_text()):.2f}" for label in ax.get_xticklabels()], rotation=45)
    ax.set_yticklabels([f"{float(label.get_text()):.2f}" for label in ax.get_yticklabels()], rotation=0)

    # Display legend
    ax.legend(fontsize=16)

    # Add labels and title
    plt.title("Impact of the Predictive Model on Re", fontsize=20)
    plt.xlabel("Free Ratio", fontsize=16)
    plt.ylabel("Isolation Ratio", fontsize=16)

    # Save the plot
    plt.tight_layout()
    plt.savefig(figure_path, dpi=500)


'''
This function computes delays, for input to the TTI framework.
This function returns three delays for the control period
'''
def compute_delays_periods(period, verbose=True):
    df = pd.read_csv(os.path.join(DATA, MERGED, '11-with_environment.csv'))[['person_id', 'test_date', 'result', 'symptoms_start_date']]

    df = df[df['result'] == 'positive']
    df = df.dropna(subset=['symptoms_start_date'])
    df['symptoms_start_date'] = pd.to_datetime(df['symptoms_start_date'])


    prereg = pd.read_csv(os.path.join(DATA, RAW, 'preregistration.csv'))[['person_id', 'test_date', 'prereg_submit_time', 'test_time']]

    prereg['prereg_submit_time'] = pd.to_datetime(prereg['prereg_submit_time'])
    prereg['test_time'] = pd.to_datetime(prereg['test_time'])

    labres = pd.read_csv(os.path.join(DATA, RAW, 'godata_labres.csv'))[['person_id', 'test_date', 'test_result_avail_time']]
    labres['test_result_avail_time'] = pd.to_datetime(labres['test_result_avail_time'])

    is_within_period = (df['test_date'] >= period[0]) & (df['test_date'] <= period[1])
    the_df = df[is_within_period]

    the_df = pd.merge(the_df, prereg, how='left')
    the_df = pd.merge(the_df, labres, how='left')

    the_df = the_df.dropna(subset=['prereg_submit_time'])
    the_df = the_df.dropna(subset=['test_result_avail_time'])

    the_df['delay_symptoms_test'] = (the_df['test_time'] - the_df['symptoms_start_date']).dt.total_seconds() / (3600 * 24)  # in days

    is_outlier = (the_df['delay_symptoms_test'] < 0) | (the_df['delay_symptoms_test'] > 14)
    the_df = the_df[~is_outlier]

    # Calculate delays in hours or days
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

    if verbose:
        print(period)
        print(f"Mean delay between symptoms onset and taking an appointment: {mean_delay_symptoms_prereg:.2f} days")
        print(f"Mean delay between taking an appointment and sampling: {mean_delay_prereg_test:.2f} days")
        print(f"Mean delay between sampling and result reception: {mean_delay_test_reception:.2f} days")

    return mean_delay_symptoms_prereg, mean_delay_prereg_test, mean_delay_test_reception


'''
This function computes the average number of daily infections during @period.
'''
def get_number_infections(period):
    df = pd.read_csv(os.path.join(DATA, RAW, 'all_tests.csv'))

    n_pos = df[(df['test_date'] >= period[0]) & (df['test_date'] <= period[1]) & (df['result'] == 'positive')].shape[0]

    start = datetime.strptime(period[0], "%Y-%m-%d")
    end = datetime.strptime(period[1], "%Y-%m-%d")

    days_elapsed = (end - start).days +1 # Because we consider both the first and last day of period.

    return n_pos/days_elapsed


'''
This function gets the incubation period, based on the mode value during a period specified period of time.
'''
def get_incubation_period(period):
    df = pd.read_csv(os.path.join(DATA, SECONDARY, 'delays.csv'))

    is_within_period = (df['week_start'] >= period[0]) & (df['week_start'] <= period[1])

    df = df[is_within_period]

    return df['incubation_time'].mode()[0]


'''
Returns the sensitivity corresponding to @metric = @metric_values
@metric is any string in ['fpr', '1-fpr', '1-sens'].
'''
def get_sensitivity(metrics_path, metric, metric_value):
    metrics = pd.read_csv(metrics_path)

    sens_lookup = metrics.drop_duplicates(metric).set_index(metric)['sensitivity']

    closest_fpr = sens_lookup.index[np.abs(sens_lookup.index - metric_value).argmin()]
    sens_val = sens_lookup.loc[closest_fpr]

    return sens_val


def get_stochastic_re(FPR_i, FNR_r, period):
    R_init = get_observed_re(period[0])

    sa, at, tr = compute_delays_periods(period=period, verbose=False)
    ip = get_incubation_period(period=period)

    # n_inf: number of infected individuals during CONTROL_TEST_PERIOD
    n_inf = get_number_infections(period=period)

    sens_i_R = get_sensitivity(os.path.join(DATA, IMPACT_EPI, 'control_metrics.csv'),
                             'fpr', FPR_i)
    # FNR = 1 - Sensitivity. We must still find the closest existing value within the model metrics though.
    sens_r_R = get_sensitivity(os.path.join(DATA, IMPACT_EPI, 'control_metrics.csv'),
                             '1-sens', FNR_r)

    res = compute_re("XX", sens_i_R, sens_r_R, R_init, sa, at, tr, ip, stoch=True, n_inf=int(n_inf))

    print(f"Re [Q1, mean, Q3]: [{res[1]:.2f}, {res[0]:.2f}, {res[2]:.2f}]")

    return res


def visualise_epi_impact(matrix_metrics, metrics_path, figure_path, testing_period):
    df = pd.read_csv(matrix_metrics, index_col=0)
    metrics = pd.read_csv(metrics_path)

    # Prepare X and Y coordinates from column and index names
    x_coords = []
    y_coords = []
    values = []

    for y_val in df.index:
        for x_val in df.columns:
            value = df.at[y_val, x_val]  # ← no float()
            if pd.notna(value):
                x_coords.append(float(x_val))  # For plotting
                y_coords.append(float(y_val))  # For plotting
                values.append(value)

    # Define colors and cmap
    v_max = max(values)
    v_min = min(values)

    if v_min <= 1.0:
        one_position = (1.0 - v_min) / (v_max - v_min)
        cmap = LinearSegmentedColormap.from_list(
            "green_white_red",
            [
                (0.0, "#519C6A"),
                (one_position, "white"),  # <-- exactly at value 1.0
                (1.0, "#9c0202"),
            ]
        )
        norm = Normalize(vmin=v_min, vmax=v_max)
    else:
        cmap = LinearSegmentedColormap.from_list(
            "green_white_red",
            [
                (0.0, "white"),  # <-- exactly at value 1.0
                (1.0, "#9c0202"),
            ]
        )
        norm = Normalize(vmin=1.0, vmax=v_max)

    tick_fontsize = 24
    marker_color = DARK_GRAY

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(x_coords, y_coords, c=values, cmap=cmap, norm=norm, s=100)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.2, aspect=8, pad=0.20)  # ← Increase pad to move it further right
    cbar.set_label('Effective Reproduction Number', fontsize=tick_fontsize)
    cbar.outline.set_visible(False)

    ax.set_xlabel(r'$SENS_{isolation}$', fontsize=tick_fontsize)
    ax.set_ylabel(r'$FNR_{release}$', fontsize=tick_fontsize)


    # DEFINE TOP AXIS
    xticks = ax.get_xticks()
    bottom_labels = [f"{i:.2f}" for i in xticks]
    fpr_lookup = metrics.drop_duplicates('sensitivity').set_index('sensitivity')['fpr']
    top_labels = []
    for val in xticks:
        try:
            closest_sens = fpr_lookup.index[np.abs(fpr_lookup.index - val).argmin()]
            fpr_val = fpr_lookup.loc[closest_sens]
            top_labels.append(f"{fpr_val:.2f}")
        except:
            top_labels.append("")

    ax_top = ax.twiny()
    xlim = ax.get_xlim()

    # The following 3 instructions should be executed for both complementary axes.
    # By doing so, the tick labels on the top and the bottom match.
    # NB: Similarly, we could have used : ax.callbacks.connect('xlim_changed', lambda ax: ax_top.set_xlim(xlim))
    ax.set_xticks(xticks) # Prevent further operations to change xticks.
    ax.set_xticklabels(bottom_labels)
    ax.set_xlim(xlim)

    ax_top.set_xticks(xticks)
    ax_top.set_xticklabels(top_labels)
    ax_top.set_xlim(xlim)

    ax_top.set_xlabel(r'$FPR_{isolation}$', fontsize=tick_fontsize)


    # DEFINE RIGHT AXIS
    yticks = ax.get_yticks()
    left_labels = [f"{i:.2f}" for i in yticks]
    one_minus_fpr_lookup = metrics.drop_duplicates('1-sens').set_index('1-sens')['1-fpr']
    right_labels = []
    for val in yticks:
        try:
            closest_fpr = one_minus_fpr_lookup.index[np.abs(one_minus_fpr_lookup.index - val).argmin()]
            right_val = one_minus_fpr_lookup.loc[closest_fpr]
            right_labels.append(f"{right_val:.2f}")
        except:
            right_labels.append("")

    ax_right = ax.twinx()
    ylim = ax.get_ylim()

    # The following 3 instructions should be executed for both complementary axes.
    # By doing so, the tick labels on the top and the bottom match.
    # NB: Similarly, we could have used : ax.callbacks.connect('ylim_changed', lambda ax: ax_right.set_ylim(ylim))
    ax.set_yticks(yticks) # Prevent further operations to change xticks.
    ax.set_yticklabels(left_labels)
    ax.set_ylim(ylim)

    ax_right.set_yticks(yticks)
    ax_right.set_yticklabels(right_labels)
    ax_right.set_ylim(ylim)

    ax_right.set_ylabel(r'$TNR_{release}$', fontsize=tick_fontsize)

    # fpr1 = 0.03
    # fnr1 = simulate_scenario_1(FPR_max=fpr1, FNR_min=None, T_max=500)

    df = df.iloc[1:, :]

    x = 0.1
    pad = (1.0-x-x)/3
    for fpr, fnr, strat, marker in zip([0.00, 0.00, 0.25, 0.10],
                                       [0.00, 0.80, 0.10, 0.35],
                                       TRIAGE_GOALS,
                                       ['*', 'o', '^', 's']):
        res = get_stochastic_re(fpr, fnr, testing_period)

        # Draw the dot on the heatmap.
        sens_lookup = metrics.drop_duplicates('fpr').set_index('fpr')['sensitivity']

        closest_fpr = sens_lookup.index[np.abs(sens_lookup.index - fpr).argmin()]
        sens_val = sens_lookup.loc[closest_fpr]

        ax.plot(sens_val, fnr, marker, color=marker_color, markersize=10, label=strat)

        # Draw the dot on the color scale.
        closest_fnr = df.index.to_series().sub(fnr).abs().idxmin()

        sens_cols = df.columns.astype(float)
        closest_sens = sens_cols[np.abs(sens_cols - sens_val).argmin()]

        r_val = df.loc[closest_fnr, str(closest_sens)]

        # Check that the mean obtained with stochastic process is similar than the R value obtained without,
        # since the values within the triangle matrix are computed without stochasticity.
        print(f"{strat}: {r_val:.2f}")
        mean, q1, q3 = res

        lower_err = mean - q1
        upper_err = q3 - mean
        cbar.ax.errorbar(
            x,
            mean,
            yerr=[[lower_err], [upper_err]],
            fmt=marker,  # triangle marker
            color=marker_color,  # marker color
            ecolor=marker_color,  # error bar color
            markersize=8,
            capsize=4
        )
        # cbar.ax.scatter(0.5, r_val, marker=marker, color=marker_color, s=50, clip_on=False)

        x = x+pad

    # Set font size
    ax.tick_params(axis='both', labelsize=18)
    ax_top.tick_params(axis='x', labelsize=18)
    ax_right.tick_params(axis='y', labelsize=18)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    ax.legend(loc='upper right', fontsize=16, title='Triage goal', title_fontsize=16)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=600)


'''
This function trains an XGBoost model, with validated hyperparameters.
@train_period refers to the period which we consider to build the training data set.
@test_period refers to the period which we consider to build the testing data set.
We assume that the testing period starts on the day after the training period.
'''
def train_model_control(train_period, test_period, suffix=''):
    df = pd.read_csv(os.path.join(DATA, PREPROCESSED, 'preprocessed.csv'))

    df = df[df['test_criterion'] != 'positive_self_test']

    is_within_period = (df['test_date'] >= train_period[0]) & (df['test_date'] <= test_period[1])
    df = df[is_within_period]

    model = joblib.load(os.path.join(DATA, TRAINED_MODELS, 'whole_xgboost_model.pkl'))
    best_params = model.get_params()

    week_start_values = df['test_date'].values
    df_pp = preprocess(df)
    df_pp = scale_data(df_pp)
    df_pp.insert(loc=1, column='test_date', value=week_start_values)

    df_train = df_pp[df_pp['test_date'] < train_period[1]]
    df_test = df_pp[df_pp['test_date'] >= train_period[1]]

    print(f"Training set contains {df_train.shape[0]} observations.")
    print(f"Testing set contains {df_test.shape[0]} observations.")

    df_train = df_train.drop('test_date', axis=1)
    df_test = df_test.drop('test_date', axis=1)

    X_train = df_train.iloc[:, 1:]
    X_test = df_test.iloc[:, 1:]
    y_train = df_train.iloc[:, 0]
    y_test = df_test.iloc[:, 0]

    model = XGBClassifier(**best_params)

    model.fit(X_train, y_train)

    y_scores = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_scores)
    print(f"ROC AUC: {auc}")

    predictions = pd.DataFrame({'actual': y_test, 'predicted': y_scores})
    predictions.to_csv(os.path.join(DATA, IMPACT_EPI, f'control_predictions{suffix}.csv'), index=False)

    with open(os.path.join(DATA, IMPACT_EPI, f"control_X_test{suffix}.pkl"), "wb") as file:
        pickle.dump(X_test, file)

    with open(os.path.join(DATA, IMPACT_EPI, f"control_model{suffix}.pkl"), "wb") as file:
        pickle.dump(model, file)


'''
This function executes R code to compute the matrix presenting the influence of decision thresholds on the reproductive number.
'''
def compute_re_matrix(sensitivity_file, output_file, period):
    R_init = get_observed_re(period[0])

    sa, at, tr = compute_delays_periods(period=period, verbose=False)
    ip = get_incubation_period(period=period)

    command = [
        "Rscript",
        "compute_Re_matrix.R",
        str(R_init),
        str(sa),
        str(at),
        str(tr),
        str(ip),
        sensitivity_file,
        output_file
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.stderr:
        print(result.stderr)


if __name__ == '__main__':
    # EVALUATE THE EPIDEMIC IMPACT OF THE TRIAGE SYSTEM OVER DIFFERENT EPIDEMIC PERIODS.

    for p_train, p_test, suffix in zip(training_periods, testing_periods, suffixes):
        print(f"PERIOD : {suffix}")

        train_model_control(p_train, p_test, suffix)
        print(get_observed_re(p_test[0]))

        # sensitivity_vector(os.path.join(DATA, IMPACT_EPI, f'control_predictions{suffix}.csv'),
        #                    os.path.join(DATA, IMPACT_EPI, f'control_sensitivity{suffix}.csv'))
        #
        # metrics_vectors(os.path.join(DATA, IMPACT_EPI, f'control_predictions{suffix}.csv'),
        #                 os.path.join(DATA, IMPACT_EPI, f'control_metrics{suffix}.csv'))
        #
        # compute_re_matrix(os.path.join(DATA, IMPACT_EPI, f'control_sensitivity{suffix}.csv'),
        #                   os.path.join(DATA, IMPACT_EPI, f'control_re_matrix{suffix}.csv'),
        #                   p_test)

        # transform_matrix(os.path.join(DATA, IMPACT_EPI, f'control_re_matrix{suffix}.csv'),
        #                  os.path.join(DATA, IMPACT_EPI, f'control_re_matrix_transformed{suffix}.csv'))

        # visualise_epi_impact(
        #     os.path.join(DATA, IMPACT_EPI, f'control_re_matrix_transformed{suffix}.csv'),
        #     os.path.join(DATA, IMPACT_EPI, f'control_metrics{suffix}.csv'),
        #     os.path.join(FIGURES, IMPACT_EPI, f'heatmap_mixed_parameters{suffix}.png'),
        #     testing_period=p_test
        # )

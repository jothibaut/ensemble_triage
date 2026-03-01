'''
Within this script, we study the impact of implementing the machine learning model in real-life on testing requirements.
'''

from commons import *

import numpy as np

from shared import *

import matplotlib.pyplot as plt


def thresholds_tests_matrix_triangle(predictions_path, metrics_path, matrix_metrics_output):
    predictions = pd.read_csv(predictions_path)
    p_infected = predictions['actual'].mean()

    print(f'{p_infected*100:.2f}% of infected patients within the data set.')

    metrics = pd.read_csv(metrics_path)

    metrics = metrics.iloc[0:].reset_index(drop=True)
    metrics = metrics.sort_values(by='threshold', ascending=False)

    # Get unique column/row labels
    sens_values = metrics['sensitivity'].unique()
    one_minus_sens_values = metrics['1-sens'].unique() # FNR

    # Initialize a DataFrame to hold the results
    matrix_metr = pd.DataFrame(index=sorted(one_minus_sens_values, reverse=True), columns=sorted(sens_values))

    for i in range(metrics.shape[0]): # isolation index
        if i % 100 == 0:
            print(f'Iteration column {i}')
        for m in range(i, metrics.shape[0]): # missed index
            sens_m = metrics.iloc[m]['sensitivity']
            sens_i = metrics.iloc[i]['sensitivity']
            fpr_m = metrics.iloc[m]['fpr']
            fpr_i = metrics.iloc[i]['fpr']

            n_infected = sens_m - sens_i # Number of tests for infected population.
            n_healthy = fpr_m - fpr_i # Number of tests for healthy population.

            n = n_infected * p_infected + n_healthy * (1-p_infected) # Total number of tests, as a proportion.
            n_saved = 1 - n # Total number of saved tests, as a proportion.

            # Here, since two different thresholds can have the same FPR|sensitivity value, we sometimes overwrite previous value.
            # Therefore, this matrix is not squared.
            col_key = sens_i
            row_key = metrics.iloc[m]['1-sens']
            matrix_metr.at[row_key, col_key] = n_saved

    matrix_metr.to_csv(matrix_metrics_output)


def visualise_tests_impact_triangle(matrix_metrics, metrics_path, predictions_path, figure_path, cmap='viridis'):
    df = pd.read_csv(matrix_metrics, index_col=0)
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    p_infected = predictions['actual'].mean()

    # Prepare X and Y coordinates from column and index names
    x_coords = []
    y_coords = []
    values = []

    for y_val in df.index: # FNR
        for x_val in df.columns: # SENS
            value = df.at[y_val, x_val]  # ← no float()
            if pd.notna(value):
                x_coords.append(float(x_val))  # For plotting
                y_coords.append(float(y_val))  # For plotting
                values.append(value)

    tick_fontsize = 16  # or whatever size you want
    marker_color = DARK_GRAY

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(x_coords, y_coords, c=values, cmap=cmap, s=50)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.12)  # ← Increase pad to move it further right
    cbar.set_label('Proportion of tests saved', fontsize=tick_fontsize)
    cbar.outline.set_visible(True)

    # ax.set_xlabel(r'$Sensitivity=\frac{isolated&infected}{infected}$')
    # ax.set_ylabel(r'$\frac{exempt&healthy}{healthy}=Specificity-\frac{tested&healthy}{healthy}$')
    ax.set_xlabel(r'$SENS_{isolation}$', fontsize=tick_fontsize)
    ax.set_ylabel(r'$FNR_{release}$', fontsize=tick_fontsize)

    ax.set_title('Impact of the model on tests saved', fontsize=tick_fontsize)

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

    ax.set_xticks(xticks)
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

    ax.set_yticks(yticks)
    ax.set_yticklabels(left_labels)
    ax.set_ylim(ylim)

    ax_right.set_yticks(yticks)
    ax_right.set_yticklabels(right_labels)
    ax_right.set_ylim(ylim)

    ax_right.set_ylabel(r'$TNR_{release}$', fontsize=tick_fontsize)

    for fpr_isolation, fnr_release, strat, marker in zip([0.00, 0.00, 0.25, 0.10],
                                                         [0.00, 0.80, 0.10, 0.35],
                                                         TRIAGE_GOALS,
                                                         ['*', 'o', '^', 's']):
        print(strat)
        fpr_isolation_idx = np.argmin(np.abs(metrics['fpr'].values - fpr_isolation))

        sensitivity_isolation = metrics.loc[fpr_isolation_idx, 'sensitivity']

        fnr_release_idx = np.argmin(np.abs(metrics['1-sens'].values - fnr_release))
        sensitivity_release = 1-fnr_release
        fpr_release = metrics.loc[fnr_release_idx, 'fpr']


        n_infected = sensitivity_release - sensitivity_isolation  # Number of tests for infected population.
        n_healthy = fpr_release - fpr_isolation  # Number of tests for healthy population.

        n = n_infected * p_infected + n_healthy * (1 - p_infected)  # Total number of tests, as a proportion.
        n_saved = 1 - n  # Total number of saved tests, as a proportion.

        if strat == 'At risk testing':
            print(f"n+ = {sensitivity_release:.2f} - {sensitivity_isolation:.2f} = {n_infected:.2f}")
            print(f"n- = {fpr_release:.2f} - {fpr_isolation:.2f} = {n_healthy:.2f}")
            print(f"n = {n_infected:.2f} * {p_infected:.2f} + {n_healthy:.2f} * {1 - p_infected:.2f} = {n}")
            print(f"({fpr_isolation}; {fnr_release}): {n}")
        print(f'Proportion of tests saved: {n_saved:.2f}')


        ax.plot(sensitivity_isolation, fnr_release, marker, color=marker_color, markersize=10, label=strat)
        cbar.ax.scatter(0.5, n_saved, marker=marker, color=marker_color, s=50, clip_on=False)

    # Set font size
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax_top.tick_params(axis='x', labelsize=tick_fontsize)
    ax_right.tick_params(axis='y', labelsize=tick_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    ax.legend(loc='upper right', fontsize=16, title='Triage goal', title_fontsize=16)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)

'''
This scripts visualises the evolution of the  vaccination status in the student population in Leuven.
'''

from commons import *

import pandas as pd


def visualise_vaccination_coverage(data_path, figure_path):
    df = pd.read_csv(data_path)

    df = df[df['week_start'] >= '2020-11-16']

    fontsize = 18
    ticksize = 16

    col_line = BORDEAUX
    col_bars = DARK_GRAY

    fig, ax1 = plt.subplots(figsize=(16, 6))

    # Plot the vaccination_rate on the left y-axis
    ax1.plot(df['week_start'], df['vaccination_rate'], color=col_line, marker='o', label='Vaccination Rate')

    ax1.set_xlabel('Week', fontsize=fontsize)
    ax1.set_ylabel('Vaccination Rate', fontsize=fontsize, color=col_line)
    ax1.tick_params(axis='y', labelsize=ticksize, colors=col_line)
    ax1.set_ylim(bottom=0)

    # X-axis ticks (every 4th week)
    xticks = df['week_start'][::4]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=ticksize)

    # Create a second y-axis for the stacked bar chart
    ax2 = ax1.twinx()

    df = df.rename(columns={'one_dose': 'One dose',
                            'two_doses': 'Two doses',
                            'unvaccinated': 'Unvaccinated'})

    # Vaccine type columns
    vaccine_cols = ['Unvaccinated', 'One dose', 'Two doses']
    # colors = ['#4C72B0', '#55A868', '#C44E52']
    colors = [GRAY, INDIGO, SALMON]

    # Plot stacked bars on the right y-axis
    bottom = pd.Series([0] * len(df))

    for idx, col in enumerate(vaccine_cols):
        ax2.bar(df['week_start'], df[col], bottom=bottom, label=col, color=colors[idx], alpha=0.4)
        bottom += df[col]

    ax2.set_ylabel('Number of reported status', fontsize=fontsize, color=col_bars)
    ax2.tick_params(axis='y', labelsize=ticksize, colors=col_bars)

    # Combine legends
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(lines, labels, loc='upper left')

    plt.title('Vaccination Rate Over Time', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)


def visualise_vaccination_delays(data_path, figure_path):
    df = pd.read_csv(data_path)

    df = df[df['week_start'] >= '2020-11-16']

    fontsize = 18
    ticksize = 16

    # Ensure dates are in datetime format
    df['week_start'] = pd.to_datetime(df['week_start'])

    # Sort for plotting
    df = df.sort_values(['vaccination_status', 'week_start'])

    # Get the complete list of week_start dates
    all_dates = pd.date_range(start=df['week_start'].min(), end=df['week_start'].max(), freq='W-MON')

    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))

    for vaccine, group in df.groupby('vaccination_status'):
        if vaccine == 'One dose':
            color = INDIGO
        elif vaccine == 'Two doses':
            color = SALMON
        else:
            color = None
        group = group.set_index('week_start').reindex(all_dates).reset_index()
        group.rename(columns={'index': 'week_start'}, inplace=True)

        plt.plot(group['week_start'], group['mean_delay'], label=vaccine, color=color)
        plt.fill_between(group['week_start'], group['lower_bound'], group['upper_bound'], alpha=0.2, color=color)

    xticks = df['week_start'].drop_duplicates().sort_values()[::4]  # every 4th week

    ax.set_xticks(xticks)  # set the positions
    ax.set_xticklabels(xticks.dt.strftime('%Y-%m-%d'), rotation=45, ha='right', fontsize=ticksize)
    ax.tick_params(axis='y', labelsize=ticksize)

    plt.xlabel('Week Start', fontsize=fontsize)
    plt.ylabel('Average Vaccination Delay (days)', fontsize=fontsize)
    plt.title('Average Vaccination Delay Over Time by Vaccine Type', fontsize=fontsize)
    plt.legend(title='Vaccination status', title_fontsize=ticksize, fontsize=ticksize, loc='upper left')
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
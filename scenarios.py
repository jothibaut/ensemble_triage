'''
Within this file, we simulate three different epidemiological scenarios,
and study how our model would perform for each of them.
'''

from commons import *
from whole_period import add_epi_impact, add_epi_impact_list, visualise_ml_impact_evolution

import os
import pandas as pd


# Metrics for mixed parameters control scenario
FPR_stable, FNR_stable = 0.00, 0.80
FPR_surges, FNR_surges = 0.25, 0.10
FPR_decr, FNR_decr = 0.10, 0.35
FPR_I = [FPR_stable, FPR_surges, FPR_decr, FPR_stable, FPR_surges, FPR_decr]
FNR_R = [FNR_stable, FNR_surges, FNR_decr, FNR_stable, FNR_surges, FNR_decr]
history = ['2020-11-16', '2021-02-08', '2021-05-10', '2021-06-07', '2021-09-13', '2022-02-21', '2022-05-30']
phases = ['stable', 'surges', 'decreasing', 'stable', 'surges', 'decreasing']


def limited_testing_capacity_scenario(output_data_file, figure_file=None, compute_data=True):
    print('LIMITED TESTING CAPACITY SCENARIO')
    T_max = 500
    if compute_data:
        result_df = pd.read_csv(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, 'data_ml.csv'))
        result_df['week_start'] = pd.to_datetime(result_df['week_start'])

        result_df = add_epi_impact(result_df, FPR_max=0.03, FNR_min=None, T_max=T_max, FNR_max=0.1)

        result_df.to_csv(output_data_file, index=False)

    if figure_file is not None:
        visualise_ml_impact_evolution(output_data_file,
                                      figure_file,
                                      f'Scenario 1: Limited testing capacity ({T_max} tests/week)')


def societal_commitment_scenario(output_data_file, figure_file=None, compute_data=True):
    print('SOCIETAL COMMITMENT SCENARIO')
    FPR_max = 0.8
    if compute_data:
        result_df = pd.read_csv(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, 'data_ml.csv'))
        result_df['week_start'] = pd.to_datetime(result_df['week_start'])

        result_df = add_epi_impact(result_df, FPR_max=FPR_max, FNR_min=0.0, T_max=None, FNR_max=0.0)

        result_df.to_csv(output_data_file, index=False)

    if figure_file is not None:
        visualise_ml_impact_evolution(output_data_file,
                                      figure_file,
                                      f'Scenario 2: High societal compliance (Isolating {int(FPR_max*100):d}% of infected cases)')


def surveillance_scenario(output_data_file, figure_file=None, compute_data=True):
    print('SURVEILLANCE SCENARIO')
    FNR = 0.95
    if compute_data:
        result_df = pd.read_csv(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, 'data_ml.csv'))
        result_df['week_start'] = pd.to_datetime(result_df['week_start'])

        result_df = add_epi_impact(result_df, FPR_max=0.0, FNR_min=FNR, T_max=None, FNR_max=FNR)

        result_df.to_csv(output_data_file, index=False)

    if figure_file is not None:
        visualise_ml_impact_evolution(output_data_file,
                                      figure_file,
                                      f'Scenario 3: Epidemiological surveillance (Testing {int((1-FNR)*100):d}% of infected cases)')


def control_scenario(output_data_file, figure_file=None, compute_data=True):
    print('CONTROL SCENARIO')
    if compute_data:
        result_df = pd.read_csv(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, 'data_ml.csv'))
        result_df['week_start'] = pd.to_datetime(result_df['week_start'])

        result_df = add_epi_impact(result_df, FPR_max=0.8, FNR_min=0.0, T_max=None, FNR_max=0.0)


        result_df.to_csv(output_data_file, index=False)

    if figure_file is not None:
        visualise_ml_impact_evolution(output_data_file,
                                      figure_file,
                                      f'Scenario X: Outbreak control')


def mixed_control_scenario(output_data_file, figure_file=None, compute_data=True):
    print('MIXED CONTROL SCENARIO')
    if compute_data:
        result_df = pd.read_csv(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, 'data_ml.csv'))

        result_df = add_epi_impact_list(result_df,
                                        FPR_max_list = FPR_I,
                                        FNR_min_list = FNR_R,
                                        FNR_max_list = FNR_R,
                                        history_list = history,
                                        phases_list = phases)

        result_df.to_csv(output_data_file, index=False)

    if figure_file is not None:
        visualise_ml_impact_evolution(output_data_file,
                                      figure_file,
                                      f'Example of a triage strategy to help control an outbreak')


if __name__ == '__main__':
    # limited_testing_capacity_scenario(os.path.join(DATA, SCENARIOS, '1_limited_testing.csv'),
    #                                   os.path.join(FIGURES, SCENARIOS, "1_limited_testing.svg"),
    #                                   compute_data=True)
    # societal_commitment_scenario(os.path.join(DATA, SCENARIOS, '2_societal_commitment.csv'),
    #                              os.path.join(FIGURES, SCENARIOS, '2_societal_commitment.svg'),
    #                              compute_data=True)
    # surveillance_scenario(os.path.join(DATA, SCENARIOS, '3_surveillance.csv'),
    #                       os.path.join(FIGURES, SCENARIOS, '3_surveillance.svg'),
    #                       compute_data=True)
    mixed_control_scenario(os.path.join(DATA, SCENARIOS, '4_mixed.csv'),
                           figure_file=os.path.join(FIGURES, SCENARIOS, '4_mixed.svg'),
                           compute_data=True)

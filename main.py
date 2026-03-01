'''
This file will be the main pipeline for the project.
'''
from impact_epi import visualise_epi_impact
from impact_tests import thresholds_tests_matrix_triangle, visualise_tests_impact_triangle
from preprocessing import preprocessing, discretise_data, one_hot_encode
from ensemble_techniques import simplififed_preprocess_for_ncv, execute_nested_cv, display_metrics, MODEL_PARAMS
from whole_period import epi_impact_evolution_data
from scenarios import *
from bayesian_networks import bn_discrete
from vaccination import *
from whole_period import visualise_relevant_questions_evolution


def display_all_models_metrics():
    for model_name, _ in MODEL_PARAMS.items():
        print(f'Performance of {model_name} model')
        display_metrics(os.path.join(DATA, SECONDARY, f'whole_{model_name}_predictions.csv'))
        print('\n')


if __name__ == '__main__':

    # The commented lines of code require access to a database with all personal information.
    # Our data set was not included to respect the privacy of participants.
    # An example, with non-real data is included in data/merged/outcome_features_example.csv.

    # preprocessing(do_period_split=False)
    #
    # simplififed_preprocess_for_ncv(os.path.join(DATA, PREPROCESSED, 'preprocessed.csv'),
    #                                WHOLE_PERIOD,
    #                                os.path.join(DATA, PREPROCESSED, 'whole_imputed.csv'),
    #                                os.path.join(DATA, PREPROCESSED, 'whole_for_ncv.csv'))
    # execute_nested_cv(os.path.join(DATA, PREPROCESSED, 'whole_for_ncv.csv'), 'whole')
    # display_all_models_metrics()

    # We chose the XGBoost model.
    # metrics_vectors(os.path.join(DATA, SECONDARY, 'whole_xgboost_predictions.csv'),
    #                 os.path.join(DATA, SECONDARY, 'whole_xgboost_metrics.csv'))
    #
    #
    # epi_impact_evolution_data()

    mixed_control_scenario(os.path.join(DATA, SCENARIOS, '4_mixed.csv'),
                           figure_file=os.path.join(FIGURES, SCENARIOS, '4_mixed.svg'),
                           compute_data=False)

    visualise_vaccination_coverage(os.path.join(DATA, VACCINATION, 'vaccination_coverage.csv'),
                                   os.path.join(FIGURES, VACCINATION, 'vaccination_coverage.svg'))

    visualise_vaccination_delays(os.path.join(DATA, VACCINATION, 'vaccination_delays.csv'),
                                 os.path.join(FIGURES, VACCINATION, 'vaccination_delays.svg'))

    visualise_relevant_questions_evolution(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, 'data_ml.csv'),
                                           os.path.join(FIGURES, WHOLE_PERIOD_ANALYSIS, 'questions_evolution.svg'))

    # visualise_epi_impact(
    #     os.path.join(DATA, IMPACT_EPI, f'control_re_matrix_transformed.csv'),
    #     os.path.join(DATA, IMPACT_EPI, f'control_metrics.csv'),
    #     os.path.join(FIGURES, IMPACT_EPI, f'heatmap_mixed_parameters.png'),
    #     testing_period=p_test
    # )

    # SUPPLEMENTARY_INFO
    # Interactive triage system:
    # run interactive_triage_metrics.py

    # Bayesian networks
    # discretise_data(os.path.join(DATA, PREPROCESSED, 'preprocessed.csv'),
    #                 os.path.join(DATA, PREPROCESSED, 'discrete.csv'))

    # one_hot_encode(os.path.join(DATA, PREPROCESSED, 'discrete.csv'),
    #                os.path.join(DATA, PREPROCESSED, 'discrete_testcrit.csv'),
    #                'test_criterion')
    # bn_discrete(os.path.join(DATA, PREPROCESSED, 'discrete_testcrit.csv'),
    #             f'bn_discrete')

    # Test impact
    thresholds_tests_matrix_triangle(os.path.join(DATA, IMPACT_EPI, 'control_predictions.csv'),
                                     os.path.join(DATA, IMPACT_EPI, 'control_metrics.csv'),
                                     os.path.join(DATA, IMPACT_TESTS, 'control_metrics_tests_triangle.csv'))

    visualise_tests_impact_triangle(
        os.path.join(DATA, IMPACT_TESTS, 'control_metrics_tests_triangle.csv'),
        os.path.join(DATA, IMPACT_EPI, 'control_metrics.csv'),
        os.path.join(DATA, IMPACT_EPI, 'control_predictions.csv'),
        os.path.join(FIGURES, IMPACT_TESTS, 'control_tests_impact_triangle.png'),
        cmap_purple
    )



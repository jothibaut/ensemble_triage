'''
This file contains global variables.
'''

from matplotlib.colors import LinearSegmentedColormap

##################
# DATA FOLDERS
##################
DATA = 'data'
DERIVATIVES = 'derivatives'
EXAMPLE = 'example'
IMPACT_EPI = 'impact_epi'
IMPACT_TESTS = 'impact_tests'
MERGED = 'merged'
OUTPUT = 'output'
POSITIVE_SELF_TESTS = 'positive_self_tests'
PREPROCESSED = 'preprocessed'
RAW = 'raw'
SCENARIOS = 'scenarios'
SECONDARY = 'secondary'
STOCHASTIC = 'stochastic_tti'
TEST_CUMULATIVE = 'test_cumulative'
TRAINED_MODELS = 'trained_models'
TRIAGE_GROUPS = 'triage_groups'
VACCINATION = 'vaccination'
WHOLE_PERIOD_ANALYSIS = 'whole_period_analysis'

FIGURES= 'figures'
ARTICLE = 'article'
BAYESIAN_NETWORK = 'bayesian_network'
CONSTRAINT = 'constraint'
DATA_VISUALISATION = 'data_visualisation'
EXPLANATIONS = 'explanations'
METRICS_EVOLUTION = 'metrics_evolution'
MODEL_EVALUATION = 'model_evaluation'
ENSEMBLE = 'ensemble_techniques'
PATIENTS_PROFILE = 'patients_profile'
QUESTIONS = 'questions'
STATISTICS = 'statistics'
STRATIFICATION = 'stratification'
TIME_SENSITIVITY = 'time_sensitivity'
TIME_WEIGHTING = 'time_weighting'

INVESTIGATION = 'investigation'

##################
# PERIODS
##################
ALPHA = ('2021-02-01', '2021-06-27')
ALPHA_PERIOD = ('2021-02-01', '2021-10-03')
DELTA_UNVAX = ('2021-06-28', '2021-08-01')
DELTA_VAX = ('2021-12-06', '2021-12-19')

EARLY_PERIOD = ('2021-03-08', '2021-06-06')
BOOSTER_PERIOD = ('2021-12-27','2022-05-08')


# Those two periods correspond to periods during which:
#   - The BA1 (resp BA2) variant was dominant --> See 'data/WP1/85_Genomic surveillance update_21June 2022.pdf'
#   - The L2FU rate was low, which is an indicator for the quality of contact tracing.
OMICRON_PERIOD1 = ('2022-01-24', '2022-02-13')
OMICRON_PERIOD2 = ('2022-03-14', '2022-04-03')

DELTA_BA1_BA2_VAX = ('2021-12-06', '2022-05-31')

WHOLE_PERIOD = ('2020-01-01', '2022-05-31')

############################
# VARIABLES AND PARAMETERS
############################
OUTCOME = 'result'
RANDOM_STATE = 42


##########################
# COLORBLIND PALETTE
##########################
INDIGO = "#332288"
LIGHT_INDIGO = "#ADA7CF"
GREEN = "#117733"
LIGHT_GREEN = "#C2F1C8"
TURQUOISE = "#44AA99"
LIGHT_TURQUOISE = "#B4DDD6"
BLUE = "#88CCEE"
YELLOW = "#DDCC77"
SALMON = "#CC6677"
LIGHT_SALMON = "EBC2C9"
FUCHSIA = "#AA4499"
BORDEAUX = "#882255"
LIGHT_BORDEAUX = "#D197B4"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#969696"
GRAY = "#808080"
DARK_GRAY = "#3d3d3d"
BLACK = "#000000"

cmap_alpha = LinearSegmentedColormap.from_list("cmap_alpha", [BORDEAUX, FUCHSIA, SALMON,  YELLOW], N=256)
cmap_vacc = LinearSegmentedColormap.from_list("cmap_vacc", [INDIGO, GREEN, TURQUOISE, BLUE], N=256)
cmap_green = LinearSegmentedColormap.from_list("cmap_vacc", ["white", "#519C6A"], N=256)
cmap_purple= LinearSegmentedColormap.from_list("cmap_vacc", ["white", LIGHT_INDIGO], N=256)


# REDEFINITION OF THE VARIABLE NAMES
LABEL_MAP = {'sex': 'Sex',
             'age': 'Age',
             'test_criterion': 'Test reason',
             'test_criterion_positive_self_test': 'Test reason: positive self-test',
             'test_criterion_symptoms': 'Test reason: symptoms',
             'test_criterion_red_zone_test_1': 'Test reason: Back from risk country',
             'test_criterion_cluster': 'Test reason: cluster in student residence',
             'test_criterion_high_risk_test_1': 'Test reason: high risk contact test 1',
             'test_criterion_high_risk_test_2': 'Test reason: high risk contact test 2',
             'test_criterion_precaution': 'Test reason: Precaution',
             'time_since_last_positive': 'Time since last positive PCR test',
             'do_have_symptoms': 'Symptom onset',
             'symptoms_start_date': 'Time since symptom onset',
             'vaccination_status': 'Vaccination status',
             'vaccination_date': 'Time since vaccination',
             'source_person_known': 'Known infection source person',
             'source_event_known': 'Known infection source event',
             'other_unknown_attendants':'Other unknown attendants',
             'possible_superspreading': 'Possibly attended a superspreading event',
             'n_reported_contacts': 'Number of reported contacts',
             'time_since_exposure': 'Time since exposure',
             'student_residence': 'Lives in student residence',
             'other_cases_in_residence': 'Known cases in student residence',
             'shared_kitchen': 'Shared kitchen',
             'n_kitchen': 'Number of persons sharing the kitchen',
             'shared_bathroom': 'Shared bathroom',
             'n_bathroom': 'Number of persons sharing the bathroom',
             'traveller': 'Traveller',
             'travel_area': 'Travel area',
             'stopover': 'Stopover',
             'stopover_area': 'Stopover area',
             'travel_end_date': 'Time since return from travel',
             'known_cases_in_area': 'Known cases',
             'known_cases_area': 'Social context of the exposure',
             'incidence': 'Incidence',
             'incidence_7d_before': 'Recent incidence history',
             'posrate': 'Positivity rate',
             'posrate_7d_before': 'Recent positivity rate history',
             'stringency': 'Stringency index',
             'precipitations': 'Precipitations',
             'temperature': 'Temperature',
             'humidity': 'Humidity',
             'pressure': 'Pressure',
             'dI_dt': 'Derivative of incidence',
             'median_R_mean': 'Reproduction number'}

CLINICAL_FEATURES = ['result', 'time_since_last_positive', 'sex', 'age']
EPI_FEATURES = ['incidence', 'incidence_7d_before', 'posrate', 'posrate_7d_before', 'stringency']
ENVIRONMENTAL_FEATURES = ['precipitations', 'temperature', 'humidity', 'pressure']

OPTIONAL_FEATURES = ['test_criterion', 'do_have_symptoms', 'symptoms_start_date', 'is_vaccinated', 'vaccination_date',
                     'vaccination_status', 'source_event_known', 'n_reported_contacts', 'time_since_exposure',
                     'student_residence', 'traveller', 'travel_area', 'stopover', 'stopover_area', 'travel_end_date',
                     'other_cases_in_residence', 'shared_kitchen', 'n_kitchen', 'shared_bathroom', 'n_bathroom',
                     'possible_superspreading', 'known_cases_in_area', 'known_cases_area',
                     'source_person_known', 'other_unknown_attendants']

# Denomination for legends
TRIAGE_GOALS = ['Test everyone (Leuven)', 'Referral for testing', 'Isolation of high risk', 'Isolation of very high risk']

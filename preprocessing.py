'''
This script implements various data preprocessing techniques.
'''

from commons import *

import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def convert_to_int(df):
    df = df.copy()  # Avoid modifying original DataFrame

    for col in df.columns:
        if df[col].dtype == 'bool':  # Convert boolean to int
            df[col] = df[col].astype(int)

        elif df[col].dtype == 'object':  # Encode categorical variables
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    return df


def filter_out_sparse_columns(df, threshold):
    filtered_df = df.drop(columns=[col for col in df.columns if df[col].count() < threshold])

    excluded_columns = set(df.columns) - set(filtered_df.columns)
    print(f"Excluding {excluded_columns} columns which contain less than {threshold} values.")

    return filtered_df


def filter_out_sparse_rows(df):
    n1 = df.shape[0]
    # Define the columns which are (almost) always present, but are not sufficient on their own to train the model.
    anchored_columns = ['person_id', 'test_date', 'result', 'sex', 'age', 'incidence', 'incidence_7d_before', 'posrate',
                        'posrate_7d_before', 'stringency', 'precipitations', 'temperature', 'humidity', 'pressure']
    volatile_columns = list(set(df.columns) - set(anchored_columns))
    filtered_df = df[df[volatile_columns].notna().any(axis=1)]
    n2 = filtered_df.shape[0]
    print(f"{n1} - {n1-n2} = {n2} observations after excluding sparse rows.")

    return filtered_df


'''
This function removes outliers which correspond to erroneous data.
'''
def remove_error_outliers(df):
    numerical_columns = ['time_since_last_positive', 'age', 'symptoms_start_date', 'vaccination_date', 'n_reported_contacts', 'time_since_exposure',
                         'travel_end_date', 'n_kitchen', 'n_bathroom', 'incidence']

    # Remove outliers which correspond to erroneous variables, i.e. which were wrongly encoded and
    # which distribution within the data set does not represent the distribution in real-life.
    # We remove the whole row for those outliers, since those can correspond to forms with a wrong submission date.
    # Other features included within this form are probably erroneous as well.

    # Remove outliers for symptoms_start_date and travel_end_date.
    # Those intervals are considered as unrelevant if they are negative or > 14 days.
    for col in ['symptoms_start_date', 'travel_end_date']:
        is_outlier = (df[col] < 0) | (df[col] > 14)
        df = df[~is_outlier]
        print(f'Removing {is_outlier.sum()} rows with {col} values outside [0; 14] days.')

    # Age Sometimes, DOB ~= test date: Remove those values.
    is_age_outlier = df['age'] < 1
    df.loc[is_age_outlier, 'age'] = None
    # print(f'Removing {is_age_outlier.sum()} age values < 1.')

    # Shared facilities: remove n = 10,000 --> Isolated outlier, probably irrelevant.
    is_n_kitchen_outlier = df['n_kitchen'] >= 10000
    df = df[~is_n_kitchen_outlier]
    is_n_bathroom_outlier = df['n_bathroom'] >= 10000
    df = df[~is_n_bathroom_outlier]
    print(f'Removing {is_n_kitchen_outlier.sum()} rows with n_kitchen values >= 10000.')
    print(f'Removing {is_n_bathroom_outlier.sum()} rows with n_bathroom values >= 10000.')

    for col in numerical_columns:
        assert not any(df[col] < 0), f"Column '{col}' contains negative values!"

    return df


'''
This function removes the outliers which are representative of the real-world,
but belong to the tails of the distribution.
'''
def remove_real_outliers(df):
    numerical_columns = ['time_since_last_positive', 'age', 'symptoms_start_date', 'vaccination_date', 'n_reported_contacts', 'time_since_exposure',
                         'travel_end_date', 'n_kitchen', 'n_bathroom', 'incidence']

    for col in numerical_columns:
        # Calculate Q1, Q3, and IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Calculate whiskers
        lower_whisker = Q1 - 1.5 * IQR
        upper_whisker = Q3 + 1.5 * IQR
        print(f"{col} whiskers: [{lower_whisker}; {upper_whisker}]")
        is_outside_whiskers = (df[col] < lower_whisker) | (df[col] > upper_whisker)

        # Filter out values outside whiskers
        df = df[~is_outside_whiskers]
        print(f'Removing {is_outside_whiskers.sum()} observations with {col} values outside whiskers.')

    return df


'''
This function replaces all NA values within the boolean and categorical variables with their mode,
and all NA values from numeric variables with their mean or median, as specified in the method argument.
'''
def impute_missing_data(df, method):
    assert method in ['mean', 'median'], f"The {method} imputation technique is not implemented."

    # Identify column types
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    boolean_cols = df.select_dtypes(include=['bool']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    assert len(df.columns) == len(numeric_cols) + len(boolean_cols) + len(
        categorical_cols), "The number of columns split by data type does not match the total number of columns."

    # Define imputers for each type
    numeric_imputer = SimpleImputer(strategy=method)  # Mean or median for numeric variables
    boolean_categorical_imputer = SimpleImputer(strategy='most_frequent')  # Mode for categorical & boolean

    # Create a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_imputer, numeric_cols),
            ('bool_cat', boolean_categorical_imputer, boolean_cols.union(categorical_cols))
        ],
        remainder='passthrough'  # Ensure any remaining columns (if any) are kept untouched
    )

    # Apply imputation and capture the resulting column order
    df_imputed_array = preprocessor.fit_transform(df)
    transformed_column_names = list(numeric_cols) + list(boolean_cols.union(categorical_cols)) + \
                               list(preprocessor.transformers_[2][2] if 'remainder' in preprocessor.transformers_ else [])

    # Rebuild the DataFrame with proper column names
    df_imputed = pd.DataFrame(df_imputed_array, columns=transformed_column_names)

    # Convert boolean columns back if needed
    df_imputed[boolean_cols] = df_imputed[boolean_cols].astype(bool)
    df_imputed[numeric_cols] = df_imputed[numeric_cols].astype(float)

    return df_imputed


'''
This function discretises each numeric variable into 3 categories.
/!\ Limitation:
    - We split the continuous domains into three equal ranges of values --> bins.
    - Those new bins may be considerably unbalanced,
        e.g. for some variables, we would expect way more variables within the middle bin than the two others.
'''
def discretise_data(data_path, output_path, as_integer=False):
    df = pd.read_csv(data_path)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = pd.cut(df[col], bins=3, labels=['low', 'medium', 'high'])

    if as_integer:
        df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

    df.to_csv(output_path, index=False)  # Save the file without imputation methods.


'''
This function runs the general preprocessing pipeline to get the data set ready
for further statistical/ML analysis.
'''
def preprocessing(do_period_split=True):
    df = pd.read_csv(os.path.join(DATA, MERGED, 'outcome_features.csv'))

    print(f"{df.shape[0]} tests before preprocessing")

    df = remove_error_outliers(df)

    print(f"{df.shape[0]} tests without error outliers")
    df = remove_real_outliers(df)
    df = filter_out_sparse_columns(df, threshold=1000)
    df = filter_out_sparse_rows(df)

    print(f"{df.shape[0]} tests without sparse rows")

    df = df[df['test_criterion'] != 'positive_self_test']

    df.to_csv(os.path.join(DATA, PREPROCESSED, 'preprocessed.csv'), index=False)


def one_hot_encode(data_path_in, data_path_out, column):
    df = pd.read_csv(data_path_in)

    df_encoded = pd.get_dummies(df, columns=[column])

    df_encoded.to_csv(data_path_out, index=False)


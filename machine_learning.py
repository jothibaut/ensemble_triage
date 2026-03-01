'''
This script only implements a function to scale data.
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler


'''
This function scales the data frame.
'''
def scale_data(data):
    ordered_cols = data.columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    non_numeric_cols = data.select_dtypes(exclude=['float64', 'int64']).columns

    assert len(ordered_cols) > 0 and len(numeric_cols) > 0, "The automatic split between numeric/non-numeric features didn't work."

    scaler = StandardScaler()
    scaled_numeric_data = scaler.fit_transform(data[numeric_cols])

    # Convert scaled data back to DataFrame
    scaled_numeric_df = pd.DataFrame(scaled_numeric_data, columns=numeric_cols)

    # Combine scaled numeric columns and non-numeric columns
    final_df = pd.concat([scaled_numeric_df, data[non_numeric_cols].reset_index(drop=True)], axis=1)

    final_df = final_df[ordered_cols]

    return final_df
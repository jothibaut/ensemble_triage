'''
This script implements a function to train a Bayesian network based on discrete data.
'''

from commons import *

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import bnlearn as bn
import networkx as nx


def bn_discrete(path, label):
    print(f'Training BN labelled {label}...')
    df = pd.read_csv(path)
    df = shuffle(df, random_state=RANDOM_STATE)

    if 'stopover_area' in df.columns:
        df.loc[(df['stopover_area'] == 'south america') | (df['stopover_area'] == 'north america'), 'stopover_area'] = 'unknown'

    # Split dataset into train (80%) and test (20%)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)


    # Learn Bayesian Network structure from train data
    # To do so, we start from an empty network (no edge), and try to add some edges.
    # If it increases the BIC score, keep edge. Otherwise, remove it.
    # That's the Hill-Climbing algorithm.
    # BIC: How well does the model fit the data?
    # BIC = log(L) - k/2 * log(N)
    #   Higher Likelihood (L) --> Higher BIC (better fit).
    #   More parameters (k) --> Lower BIC (penalizes complexity).
    #   More data (N) --> Less penalty on complexity.
    model = bn.structure_learning.fit(train_df)

    # Parameter learning (fit CPDs)
    # Once the structure of the network is learnt, we can compute the conditional probability on each edge.
    # Write the results into a Conditional Probability Table.
    model = bn.parameter_learning.fit(model, train_df, verbose=0)

    adjmat = model['adjmat']
    graph = nx.DiGraph(adjmat)

    dotgraph = bn.plot_graphviz(model)
    dotgraph.view(filename=os.path.join(FIGURES, BAYESIAN_NETWORK, f'{label}_graph'))

    weakly_connected_components = list(nx.weakly_connected_components(graph))
    result_component = list(next(comp for comp in weakly_connected_components if OUTCOME in comp))

    # Lists to store results
    result_pred = []
    true_prob = []

    test_df_result_comp = test_df[result_component]
    # Perform inference for each row in the test set
    for i, row in test_df_result_comp.iterrows():
        evidence = row.drop(labels=[OUTCOME]).to_dict()  # Use all variables except 'bronc' as evidence
        prob_dist = bn.inference.fit(model, variables=[OUTCOME], evidence=evidence, verbose=1)

        # Get most probable value and its probability
        prob_df = prob_dist.df
        most_probable_idx = prob_df['p'].idxmax()  # Most likely value (0 or 1)
        most_probable_value = prob_df.loc[most_probable_idx, OUTCOME]
        true_probability = prob_df.loc[prob_df[OUTCOME] == True, 'p'].values[0]

        # Store results
        result_pred.append(most_probable_value)
        true_prob.append(true_probability)

    # Append results to test dataframe
    test_df['result_pred'] = result_pred
    test_df['predicted'] = true_prob

    test_df = test_df.rename(columns={'result': 'actual'})

    test_df.to_csv(os.path.join(DATA, SECONDARY, f'{label}_predictions.csv'), index=False)    # Display updated test dataframe

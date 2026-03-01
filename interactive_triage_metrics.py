'''
This script runs a small Dash app with interactive sliders to choose the isolation and release thresholds
based on probabilities.
'''

import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import os

from commons import *  # for DATA, WHOLE_PERIOD_ANALYSIS, TURQUOISE, INDIGO, SALMON

# Load master data once
df = pd.read_csv(os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, 'data_ml.csv'))
positive_tests = pd.read_csv(os.path.join(DATA, SECONDARY, 'weekly_positive_tests.csv'))

# Get list of all weeks
weeks = sorted(df["week_start"].unique())

# Cache all prediction files into a dict at startup
predictions = {}
metrics = {}
for week in weeks:
    predictions_path = os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, f"predictions_{week}.csv")
    metrics_path = os.path.join(DATA, WHOLE_PERIOD_ANALYSIS, f"metrics_{week}.csv")

    predictions[week] = pd.read_csv(predictions_path)
    metrics[week] = pd.read_csv(metrics_path)


app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.H2("Triage output"),

    # html.Label(["FPR", html.Sub("isolation"), " = "]),
    dcc.Markdown(
        r"FPR$_{isolation}$ = $\dfrac{\text{Non-infected & isolated}}{\text{Non-infected}}$",
        mathjax=True,
        style={"font-size": "18px"}
    ),

    dcc.Slider(
        id="fpri-slider",
        min=0.0,
        max=1.0,
        step=0.05,
        value=0.05,
        marks={str(round(i, 2)): str(round(i, 2)) for i in np.arange(0, 1.01, 0.1)}
    ),

    html.Br(),

    # html.Label(["FNR", html.Sub("release")]),

    dcc.Markdown(
        r"FNR$_{release}$ = $\dfrac{\text{Infected & released}}{\text{Infected}}$",
        mathjax=True,
        style={"font-size": "18px"}
    ),

    dcc.Slider(
        id="fnrr-slider",
        min=0.0,
        max=1.0,
        step=0.05,
        value=0.1,
        marks={str(round(i, 2)): str(round(i, 2)) for i in np.arange(0, 1.01, 0.1)}
    ),

    html.Br(),

    # Two stacked graphs (top: dynamic, bottom: static)
    dcc.Graph(id="triage-bar"),
])


# Callback: compute per-week triage counts
@app.callback(
    Output("triage-bar", "figure"),
    [Input("fpri-slider", "value"),
     Input("fnrr-slider", "value")]
)
def update_plot(fpri, fnrr):
    week_labels = []
    n_release_list, n_test_list, n_isolate_list = [], [], []
    pr_release_list, pr_test_list, pr_isolate_list = [], [], [] # List to store the positive rates for each category.

    for week, pred_df in predictions.items():
        metrics_df = metrics[week]
        i_thr = metrics_df.loc[metrics_df['fpr'] <= fpri, 'threshold'].values[-1]
        poss_r_thr = metrics_df.loc[metrics_df['1-sens'] <= fnrr, 'threshold']
        if len(poss_r_thr) == 0:
            r_thr = 1.0
        else:
            r_thr = metrics_df.loc[metrics_df['1-sens'] <= fnrr, 'threshold'].values[0]

        # Compute per-week triage decisions
        is_isolated = pred_df["predicted"] >= i_thr
        is_released = (pred_df["predicted"] < r_thr) & (pred_df["predicted"] < i_thr)
        is_tested = ~is_isolated & ~is_released

        n_release = is_released.sum()
        n_test = is_tested.sum()
        n_isolate = is_isolated.sum()

        pr_release = pred_df.loc[is_released, 'actual'].mean() if n_release > 0 else 0
        pr_test = pred_df.loc[is_tested, 'actual'].mean() if n_test > 0 else 0
        pr_isolate = pred_df.loc[is_isolated, 'actual'].mean() if n_isolate > 0 else 0

        week_labels.append(week)
        n_release_list.append(n_release)
        n_test_list.append(n_test)
        n_isolate_list.append(n_isolate)
        pr_release_list.append(pr_release)
        pr_test_list.append(pr_test)
        pr_isolate_list.append(pr_isolate)

    # --- Filter positive tests for visible weeks ---
    pos_df = positive_tests[positive_tests["week_start"].isin(week_labels)].copy()
    pos_df = pos_df.sort_values("week_start")

    # --- Build subplot figure ---
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.33, 0.33, 0.33],
        subplot_titles=(
            f"Triage decisions per week (FPR_i ≤ {fpri:.2f}, FNR_r ≤ {fnrr:.2f})",
            "Positive rate per triage category",
            "Weekly positive tests"
        )
    )

    # --- Top chart: triage stacked bars ---
    fig.add_trace(go.Bar(
        x=week_labels,
        y=n_release_list,
        name="Release",
        marker_color=TURQUOISE,
        opacity=0.7
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=week_labels,
        y=n_test_list,
        name="Test",
        marker_color=INDIGO,
        opacity=0.7
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=week_labels,
        y=n_isolate_list,
        name="Isolate",
        marker_color=SALMON,
        opacity=0.7
    ), row=1, col=1)

    # --- Middle chart: positive rates (lines) ---
    fig.add_trace(go.Scatter(
        x=week_labels, y=pr_release_list,
        name="Release (pos. rate)",
        mode="lines+markers",
        line=dict(color=TURQUOISE, width=2)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=week_labels, y=pr_test_list,
        name="Test (pos. rate)",
        mode="lines+markers",
        line=dict(color=INDIGO, width=2)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=week_labels, y=pr_isolate_list,
        name="Isolate (pos. rate)",
        mode="lines+markers",
        line=dict(color=SALMON, width=2)
    ), row=2, col=1)

    # --- Bottom chart: positive tests as bars ---
    fig.add_trace(go.Bar(
        x=pos_df["week_start"],
        y=pos_df["positive"],
        name="Positive tests",
        marker_color="black",
        opacity=0.6
    ), row=3, col=1)

    # --- Layout ---
    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        showlegend=True,
        legend_title="Metric",
        xaxis3=dict(title="Week", tickangle=-45),
        yaxis=dict(title="Number of individuals"),
        yaxis2=dict(title="Positive rate"),
        yaxis3=dict(title="Positive tests"),
        margin=dict(l=60, r=40, t=80, b=100),
        height=800
    )

    return fig


if __name__ == "__main__":
    app.run(debug=True)

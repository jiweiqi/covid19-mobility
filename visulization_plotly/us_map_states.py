import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import numpy as np

data_dir = '../fuel_demand_projections/'
PODA_Model = np.load(data_dir+'PODA_Model/PODA_Model_2020-06-17.npy',
                     allow_pickle='TRUE').item()

df = PODA_Model['Mobility_State_Level_Projection_mean']

year = 2020

# your color-scale
scl = [[0.0, '#ffffff'], [0.2, '#b4a8ce'], [0.4, '#8573a9'],
       [0.6, '#7159a3'], [0.8, '#5732a1'], [1.0, '#2c0579']]  # purples

fig = go.Figure()

data_slider = []
for date in df['Date'].unique():
    df_segmented = df[(df['Date'] == date)]

    for col in df_segmented.columns:
        df_segmented[col] = df_segmented[col].astype(str)

    data_each_yr = dict(
        type='choropleth',
        locations=df_segmented['State Code'],
        z=df_segmented['Apple State Mobility Predict'].astype(float),
        locationmode='USA-states',
        colorscale=px.colors.sequential.Mint,
        colorbar=dict(title='Mobility Index',
                      tickvals=[40, 60, 80, 100, 120],
                      dtick=5
                      ),
        showlegend=False,
        showscale=False)

    data_slider.append(data_each_yr)

steps = []
for i in range(len(data_slider)):

    if i % 15 == 0:
        step = dict(method='restyle',
                    args=['visible', [False] * len(data_slider)],
                    label=np.datetime_as_string(df['Date'].unique()[i], unit='D'))
        step['args'][1][i] = True
        steps.append(step)

sliders = [dict(active=0, pad={"t": 1}, steps=steps)]

layout = dict(title='Mobility Index by State',
              geo=dict(scope='usa',
                       projection={'type': 'albers usa'}),
              sliders=sliders,
              legend_orientation="h",
              autosize=True,
              height=400,
              margin=dict(l=0, r=0)
              )

fig = dict(data=data_slider, layout=layout)

pio.write_html(fig,
               file='us_map_states.html',
               config={'displayModeBar': False},
               auto_open=True,
               include_plotlyjs='cdn',
               full_html=False)

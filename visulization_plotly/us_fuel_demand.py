from datetime import timedelta

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

data_dir = '../fuel_demand_projections/'

PODA_Model = np.load(data_dir+'PODA_Model/PODA_Model_2020-07-08.npy',
                     allow_pickle='TRUE').item()

fig = go.Figure()

color_i = 'rgb(142, 212, 212)'

df_now = PODA_Model['Fuel_Demand_Projection_upper']
fig.add_trace(go.Scatter(x=df_now.index,
                         y=np.int64(df_now['Google Fuel Demand Predict']*1e3),
                         mode=None,
                         name='lower',
                         line=dict(color=color_i,
                                   width=0.0,
                                   dash=None)))

df_now = PODA_Model['Fuel_Demand_Projection_lower']
fig.add_trace(go.Scatter(x=df_now.index,
                         y=np.int64(df_now['Google Fuel Demand Predict']*1e3),
                         fill='tonexty',
                         opacity=0.2,
                         name='upper',
                         line=dict(color=color_i,
                                   width=0.0,
                                   dash=None)))

df_now = PODA_Model['Fuel_Demand_Projection_mean']
fig.add_trace(go.Scatter(x=df_now.index,
                         y=np.int64(df_now['Google Fuel Demand Predict']*1e3),
                         name='mean',
                         mode=None,
                         line=dict(color='rgb(57, 163, 163)',
                                   width=2,
                                   dash='dot')))


df_EIA = PODA_Model['Fuel_Demand_EIA'].iloc[2:]
fig.add_trace(go.Scatter(x=df_EIA.Date - timedelta(days=4),
                         y=np.int64(df_EIA['Gasoline']*1e3),
                         name='EIA',
                         mode='lines',
                         line=dict(color='darkred',
                                   width=2,
                                   dash='solid')))


fig.update_layout(title={'text': '<b>US Motor Gasoline Demand</b>',
                         'y': 0.9,
                         'x': 0.5,
                         'xanchor': 'center',
                         'yanchor': 'top'
                         },
                  font=dict(size=16, color="black")
                  )

annotations = []

fig.update_xaxes(title_text=None,
                 tickfont=dict(size=14),
                 titlefont=dict(size=16))

fig.update_yaxes(title_text='Motor gasoline demand (barrel)',
                 tickfont=dict(size=12),
                 titlefont=dict(size=14))

fig.update_layout(template='plotly_white')

fig.update_layout(xaxis=dict(tickformat="%m-%d"))
fig.update_xaxes(range=['2020-03-01', df_now.index[-1]])

fig.update_layout(annotations=annotations)

fig.update_layout(hovermode="x")
fig.update_layout(showlegend=False)

fig.update_layout(
    autosize=True,
    height=400,
    margin=dict(l=0, r=0)
)

fig.add_shape(
    dict(
        type="line",
        x0='2020-07-08',
        y0=0,
        x1='2020-07-08',
        y1=10**7,
        line=dict(
            color='rgb(65, 65, 69)',
            width=0.7
        )
    ))

fig.show()

pio.write_html(fig,
               file='us_fuel_demand.html',
               config={'displayModeBar': False},
               auto_open=True,
               include_plotlyjs='cdn',
               full_html=False)

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import numpy as np

# https://plotly.com/python/getting-started-with-chart-studio/

data_dir = '../fuel_demand_projections/'

df = pd.read_excel(data_dir+'fueldemand_2020-05-17.xlsx')

df3 = pd.read_excel(data_dir+'Fuel_Demand_3_Scenarios_2020-05-17.xlsx')

# df_EIA = pd.read_excel(data_dir+'NoPandemic_EIA2020.xlsx')

PODA_Model = np.load(data_dir+'PODA_Model/PODA_Model_2020-05-26.npy',
                     allow_pickle='TRUE').item()

fig = go.Figure()

color_i = 'rgb(36,140,140)'

df_now = PODA_Model['Fuel_Demand_Projection_upper']
fig.add_trace(go.Scatter(x=df_now['Date'],
                         y=np.int64(df_now['Apple Fuel Demand Predict']*1e3), 
                         mode=None,
                         name='lower range',
                         line=dict(color=color_i, width=0.0,
                                   dash=None)))

df_now = PODA_Model['Fuel_Demand_Projection_mean']
fig.add_trace(go.Scatter(x=df_now['Date'],
                         y=np.int64(df_now['Apple Fuel Demand Predict']*1e3), 
                         name='mean',
                         mode='lines',
                         fill='tonexty', # fill area between trace0 and trace1
                         # fillcolor='rgba(255, 0, 0, 0.1)',
                         
                         line=dict(color=color_i, width=2,
                                   dash='dash')))

df_now = PODA_Model['Fuel_Demand_Projection_lower']
fig.add_trace(go.Scatter(x=df_now['Date'],
                         y=np.int64(df_now['Apple Fuel Demand Predict']*1e3), 
                         fill='tonexty',
                         # fillcolor=color_i,
                         opacity=0.2,
                         name='upper range',
                         line=dict(color=color_i, width=0.0,
                                   dash=None)))

from datetime import timedelta
df_EIA = PODA_Model['Fuel_Demand_EIA'].iloc[2:]
fig.add_trace(go.Scatter(x=df_EIA.index - timedelta(days=7),
                         y=np.int64(df_EIA['Gasoline']*1e3), 
                         name='EIA actual',
                         mode='lines',
                         line=dict(color='darkred', 
                                   width=2,
                                   # color='rgb(27,158,119)',
                                   dash='solid')))


fig.update_layout(title={'text': '<b>US Motor Gasoline Demand</b>',
                         'y': 0.9,
                         'x': 0.5,
                         'xanchor': 'center',
                         'yanchor': 'top'
                         },
                  font=dict(size=16,
                            color="black"
                            )
                  )

annotations = []

# annotations.append(dict(xref='paper', yref='paper', x=0.0, y=0.0,
#                         xanchor='center', yanchor='top',
#                         text='EIA data was shifted 7 day to represent \
#                             the delay between actual fuel use, refueling, \
#                                 and EIA data reporting.',
#                         font=dict(size=10),
#                         showarrow=False))

fig.update_xaxes(title_text=None,
                 tickfont=dict(size=14),
                 titlefont=dict(size=16))

fig.update_yaxes(title_text='Motor gasoline demand (barrel)',
                 tickfont=dict(size=12),
                 titlefont=dict(size=14))

fig.update_layout(template='plotly_white')

fig.update_layout(xaxis=dict(tickformat="%m-%d"))

fig.update_layout(annotations=annotations)

fig.update_layout(hovermode="x")
fig.update_layout(showlegend=False)

fig.update_layout(
    autosize=True,
    # width=500,
    height=400,
    margin=dict(
        l=0,
        r=0,
        # b=0,
        # t=0,
        # pad=4
    )
)

fig.show()

pio.write_html(fig,
               file='us_fuel_demand.html',
               config={'displayModeBar': False},
               auto_open=True,
               include_plotlyjs='cdn',
               full_html=False)

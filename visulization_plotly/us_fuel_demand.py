import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import numpy as np

# https://plotly.com/python/getting-started-with-chart-studio/

df = pd.read_excel('../fueldemand_2020-05-17.xlsx')

df3 = pd.read_excel('../Fuel_Demand_3_Scenarios_2020-05-17.xlsx')

df_EIA = pd.read_excel('../NoPandemic_EIA2020.xlsx')

fig = go.Figure()

color_i = 'rgb(36,140,140)'

fig.add_trace(go.Scatter(x=df3['Date mean'],
                         y=np.int64(df3['Apple Fuel Demand upper']*1e3), 
                         mode=None,
                         name='lower range',
                         line=dict(color=color_i, width=0.0,
                                   dash=None)))

fig.add_trace(go.Scatter(x=df3['Date mean'],
                         y=np.int64(df3['Apple Fuel Demand mean']*1e3), 
                         name='mean',
                         mode='lines',
                         fill='tonexty', # fill area between trace0 and trace1
                         # fillcolor='rgba(255, 0, 0, 0.1)',
                         
                         line=dict(color=color_i, width=2,
                                   dash='dash')))

fig.add_trace(go.Scatter(x=df3['Date mean'],
                         y=np.int64(df3['Apple Fuel Demand lower']*1e3), 
                         fill='tonexty',
                         # fillcolor=color_i,
                         opacity=0.2,
                         name='upper range',
                         line=dict(color=color_i, width=0.0,
                                   dash=None)))

from datetime import timedelta
fig.add_trace(go.Scatter(x=df_EIA['Date'][9:19] - timedelta(days=7),
                         y=np.int64(df_EIA['Actual 2020'][9:19]*1e3), 
                         name='EIA actual',
                         mode='lines',
                         line=dict(color='rgb(27,158,119)', width=2,
                                   dash='solid')))


fig.update_layout(title={'text': '<b>US Motor gasoline demand</b>',
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

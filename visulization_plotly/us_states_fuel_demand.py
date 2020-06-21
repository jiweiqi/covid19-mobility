from datetime import timedelta

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

today = '2020-06-17'

data_dir = '../fuel_demand_projections/'

PODA_Model = np.load(data_dir+'PODA_Model/PODA_Model_'+today+'.npy',
                     allow_pickle='TRUE').item()

color_i = 'rgb(142, 212, 212)'

df_upper = PODA_Model['Google_Apple_Mobility_Projection_upper']
df_mean = PODA_Model['Google_Apple_Mobility_Projection_mean']
df_lower = PODA_Model['Google_Apple_Mobility_Projection_lower']

# df_label = PODA_Model['Data_for_Mobility_Projection_mean']

for i_state, state_name in enumerate(df_mean['State Name'].unique()):

    if i_state > 100:
        break

    print("ploting "+state_name + ' ' + str(i_state) + '/50')

    df_state_upper = df_upper[(df_upper['State Name'] == state_name)]
    df_state_mean = df_mean[(df_mean['State Name'] == state_name)]
    df_state_lower = df_lower[(df_lower['State Name'] == state_name)]

    for i_key, key in enumerate(df_mean.keys()):

        if i_key > 3:
            break

        fig = go.Figure()

        df_now = df_state_lower
        fig.add_trace(go.Scatter(x=df_now['date'],
                                 y=np.int64(df_now[key]),
                                 mode=None,
                                 name='lower',
                                 line=dict(color=color_i,
                                           width=0.0,
                                           dash=None)))

        df_now = df_state_upper
        fig.add_trace(go.Scatter(x=df_now['date'],
                                 y=np.int64(df_now[key]),
                                 fill='tonexty',
                                 opacity=0.2,
                                 name='upper',
                                 line=dict(color=color_i,
                                           width=0.0,
                                           dash=None)))

        df_now = df_state_mean
        fig.add_trace(go.Scatter(x=df_now['date'],
                                 y=np.int64(df_now[key]),
                                 name='mean',
                                 mode=None,
                                 line=dict(color='rgb(57, 163, 163)',
                                           width=2,
                                           dash='dot')))

        # df_EIA = df_label[(df_label['State Name'] == state_name)]
        # fig.add_trace(go.Scatter(x=df_EIA.Date - timedelta(days=4),
        #                           y=np.int64(df_EIA['Gasoline']*1e3),
        #                           name='EIA',
        #                           mode='lines',
        #                           line=dict(color='darkred',
        #                                     width=2,
        #                                     dash='solid')))

        # fig.update_layout(title={'text': '<b>'+ state_name +' Mobility Projection</b>',
        #                         'y': 0.9,
        #                         'x': 0.5,
        #                         'xanchor': 'center',
        #                         'yanchor': 'top'
        #                         },
        #                 font=dict(size=16, color="black")
        #                 )

        annotations = []

        fig.update_xaxes(title_text=None,
                         tickfont=dict(size=14),
                         titlefont=dict(size=16))

        fig.update_yaxes(title_text=key,
                         tickfont=dict(size=12),
                         titlefont=dict(size=14))

        fig.update_layout(template='plotly_white')

        fig.update_layout(xaxis=dict(tickformat="%m-%d"))
        fig.update_xaxes(range=['2020-03-01', None])

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
                x0=today,
                y0=0,
                x1=today,
                y1=5,
                line=dict(
                    color='rgb(65, 65, 69)',
                    width=0.7
                )
            ))

        # fig.show()

        pio.write_html(fig,
                       file='html/us_'+state_name.replace(" ", "_")+'_mobility_'+key.replace(" ", "_")+'.html',
                       config={'displayModeBar': False},
                       auto_open=True,
                       include_plotlyjs='cdn',
                       full_html=False)


import dash
dash.register_page(__name__,name='Download parameters', path='/download-parameters')
from dash import Dash, html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import json
from utils import AIPS_functions as af
from utils import AIPS_module as ai
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd

layout = html.Div([
                dbc.Row(html.P('Update the parameters.xml file from the parental folder:')),
                html.Br(),
                dbc.Row([
                    html.Button("Update", id="btn_update"),
                    html.Div(id='Progress',hidden=False)]),
])

@callback(
    Output('Progress', 'value'),
    [Input("btn_update", "n_clicks"),
    State('upload-image', 'filename'),
    State('act_ch','value'),
    State('block_size','value'),
    State('offset','value'),
    State('rmv_object_nuc','value'),
    State('block_size_cyto', 'value'),
    State('offset_cyto', 'value'),
    State('global_ther', 'value'),
    State('rmv_object_cyto', 'value'),
    State('rmv_object_cyto_small', 'value'),
    ])
def Save_parameters_csv(nnn,image,channel,bs,os,ron,bsc,osc,gt,roc,rocs):
    dict = {'act_ch':[channel],'block_size':[bs],'offset':[os],'rmv_object_nuc':[ron],'block_size_cyto':[bsc],
            'offset_cyto':[osc], 'global_ther':[gt],
            'rmv_object_cyto':[roc],'rmv_object_cyto_small':[rocs],'memory_reduction':1}
    df = pd.DataFrame(dict)
    df.to_csv(image[0].split('.')[0] + '_parameters.csv', encoding='utf-8', index=False)
    return [html.P("done")]
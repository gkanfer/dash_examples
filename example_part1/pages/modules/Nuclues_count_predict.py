import dash
dash.register_page(__name__, path='/Nuclues_count_predict')
from dash import Dash, html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import json
from utils import AIPS_functions as af
from utils import AIPS_module as ai
import numpy as np
from PIL import Image
import plotly.express as px

UPLOAD_DIRECTORY = "/app_uploaded_files"

layout = html.Div(
    [
        dbc.Container(
            [dbc.Row([
                dbc.Col(children=[
                    html.Div(id='output-nuc-image')
                ])
                ]),
            ]
        )
        ]
    )

@callback(
    Output('output-nuc-image','children'),
    [Input('upload-image', 'filename'),
    Input('json_img_ch', 'data'),
    Input('json_img_ch2', 'data'),
    State('act_ch', 'value'),
    State('high_pass', 'value'),
    State('low_pass', 'value'),
    State('block_size','value'),
    State('offset','value'),
    State('rmv_object_nuc','value'),
    State('block_size_cyto', 'value'),
    State('offset_cyto', 'value'),
    State('global_ther', 'value'),
    State('rmv_object_cyto', 'value'),
    State('rmv_object_cyto_small', 'value'),
     ],suppress_callback_exceptions=True)
def update_nuc(image,ch,ch2,channel,high,low,bs,os,ron,bsc,osc,gt,roc,rocs):
    memory_index = {1: [0.25, 4], 2: [0.125, 8], 3: [0.062516, 16], 4: [0.031258, 32]}
    AIPS_object = ai.Segment_over_seed(Image_name=image[0], path=UPLOAD_DIRECTORY, rmv_object_nuc=ron,
                                       block_size=bs,
                                       offset=os,
                                       block_size_cyto=bsc, offset_cyto=osc, global_ther=gt, rmv_object_cyto=roc,
                                       rmv_object_cyto_small=rocs, remove_border=False)
    ch_ = np.array(json.loads(ch))
    ch2_ = np.array(json.loads(ch2))
    ch_3c = af.gray_scale_3ch(ch_)
    nuc_s = AIPS_object.Nucleus_segmentation(ch_, rescale_image=True,scale_factor=memory_index[1])
    sort_mask = nuc_s['sort_mask']
    # segmentation traces the nucleus segmented image based on the
    fig_im_pil_sort_mask = af.plot_composite_image(ch_3c, sort_mask, fig_title='RGB map - seed', alpha=0.2)
    return [
        dbc.Row([
            dbc.Col(
                dcc.Graph(
                    id="Nuclues_pick",
                    figure=fig_im_pil_sort_mask))
            ])]

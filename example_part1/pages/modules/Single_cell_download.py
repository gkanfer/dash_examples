'''
Save single cells segmented images to local computer
'''
import dash
dash.register_page(__name__, path='/Single_cell_download')
from dash_extensions.enrich import Dash, Output, Input, State, ServersideOutput
from dash import  html, dcc, callback
from dash import callback_context,MATCH,ALL
import dash_table
import dash_bootstrap_components as dbc
import json
import tifffile as tfi
import dash_daq as daq
from skimage.exposure import rescale_intensity, histogram
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageEnhance
import base64
import pandas as pd
from io import BytesIO
from skimage import io, filters, measure, color, img_as_ubyte
from dash_extensions import Download
from dash_extensions.snippets import send_data_frame
import numpy as np
from PIL import Image
import plotly.express as px
from utils.controls import generate_team_button
from utils.Display_composit import countor_map,row_highlight
from utils import AIPS_module as ai
from utils import display_and_xml as dx
from utils import AIPS_functions as af

UPLOAD_DIRECTORY = "/download"

layout = html.Div([
    html.P('Split image to single cell images'),
    dbc.Row([
            dbc.Col(dbc.Button('map images',id='map-images',class_name='page-link',n_clicks=0),md=3),
            dbc.Col(dbc.Button('download single cell images',id='down-single-cell',class_name='page-link',n_clicks=0),md=3),
            ]),
    dcc.Dropdown(['50', '150', '200'], '150', id='cell-extract-dropdown'),
    html.P('OR'),
    dcc.Input(
            id="custom-input",
            type='text',
            placeholder="Please enter size of cell extraction",
                ),
    dcc.Loading(html.Div(id='map-display'), type="circle", style={'height': '100%', 'width': '100%'}),
    # sored items
    dcc.Store(id='dropdown_extract_size'),
    dcc.Store(id='single_cell_img_list'),
    ])

@callback(Output('dropdown_extract_size','data'),
        Input('cell-extract-dropdown','value'))
def set_extract_size(value):
    if value is None:
        return dash.no_update
    else:
        return value

@callback([
    ServersideOutput('jason_ch2', 'data'),
    ServersideOutput('json_ch2_gs_rgb', 'data'),
    ServersideOutput('json_mask_seed','data'),
    ServersideOutput('json_mask_target','data'),
    ServersideOutput('json_table_prop','data')],
   [Input('upload-image', 'filename'),
    Input('json_img_ch', 'data'),
    Input('json_img_ch2', 'data'),
    State('act_ch', 'value'),
    State('block_size','value'),
    State('offset','value'),
    State('offset_store','data'),
    State('rmv_object_nuc','value'),
    State('block_size_cyto', 'value'),
    State('offset_cyto','value'),
    State('offset_cyto_store', 'data'),
    State('global_ther', 'value'),
    State('rmv_object_cyto', 'value'),
    State('rmv_object_cyto_small', 'value'),
    State('switch_remove_border','on')],
    suppress_callback_exceptions = True,
    memoize=True)
def Generate_single_cell_image_extract(image,ch,ch2,channel,bs,os,osd,ron,bsc,osc,oscd,gt,roc,rocs,remove_bord):
    '''

    Generate
    3 channel grayscale target PIL RGB
    3 channel grayscale target PIL RGB image with seed segment
    3 channel grayscale target PIL RGB image with seed and target segment
    complete feture table
    32int seed mask
    32int target mask
    '''
    #test wheter paramters are from csv file
    if osd is None:
        os=os
    else:
        os=osd
    if oscd is None:
        osc=osc
    else:
        osc=oscd
    memory_index = {1: [0.25, 4], 2: [0.125, 8], 3: [0.062516, 16], 4: [0.031258, 32]}
    AIPS_object = ai.Segment_over_seed(Image_name=str(image[0]), path=UPLOAD_DIRECTORY, rmv_object_nuc=ron,
                                       block_size=bs,
                                       offset=os,
                                       block_size_cyto=bsc, offset_cyto=osc, global_ther=gt, rmv_object_cyto=roc,
                                       rmv_object_cyto_small=rocs, remove_border=remove_bord)
    ch_ = np.array(ch)
    ch2_ = np.array(ch2)
    # ch2_ = af.show_image_adjust(ch2_, low_prec=low, up_prec=high)
    nuc_s = AIPS_object.Nucleus_segmentation(ch_,rescale_image=True,scale_factor=memory_index[1])
    seg = AIPS_object.Cytosol_segmentation(ch_, ch2_, nuc_s['sort_mask'], nuc_s['sort_mask_bin'],rescale_image=True,scale_factor=memory_index[1])
    # segmentation traces the nucleus segmented image based on the
    ch2_255 = (ch2_ / ch2_.max()) * 255
    ch2_u8 = np.uint8(ch2_255)
    # ROI to 32bit
    sort_mask_sync = seg['sort_mask_sync']
    cseg_mask = seg['cseg_mask']
    sort_mask_sync = af.remove_gradiant_label_border(sort_mask_sync)
    cseg_mask = af.remove_gradiant_label_border(cseg_mask)
    # draw outline only
    bf_mask = dx.binary_frame_mask(ch2_u8, sort_mask_sync)
    bf_mask = np.where(bf_mask == 1, True, False)
    c_mask = dx.binary_frame_mask(ch2_u8, cseg_mask)
    c_mask = np.where(c_mask == 1, True, False)
    rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
    rgb_input_img[:, :, 0] = ch2_u8
    rgb_input_img[:, :, 1] = ch2_u8
    rgb_input_img[:, :, 2] = ch2_u8
    rgb_input_img[bf_mask > 0, 2] = 255 # 3d grayscale array where green channel is for seed segmentation
    #label_array = nuc_s['sort_mask']
    prop_names = [
        "label",
        "area",
        "eccentricity",
        "euler_number",
        "extent",
        "feret_diameter_max",
        "inertia_tensor",
        "inertia_tensor_eigvals",
        "moments",
        "moments_central",
        "moments_hu",
        "moments_normalized",
        "orientation",
        "perimeter",
        "perimeter_crofton",
        # "slice",
        "solidity"
    ]
    table_prop = measure.regionprops_table(
        cseg_mask, intensity_image=rgb_input_img, properties=prop_names
    )
    json_object_ch2 = ch2_
    json_object_ch2_seed_gs_rgb = rgb_input_img
    json_object_mask_seed = seg['sort_mask_sync']
    json_object_mask_target = seg['cseg_mask']
    json_object_table_prop = table_prop
    return json_object_ch2,json_object_ch2_seed_gs_rgb ,json_object_mask_seed,json_object_mask_target,json_object_table_prop
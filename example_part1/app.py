'''
# update branch
git add .
git commit -m "03-09-2022 correct markers only 3 errors in debug"
git branch -m server
git push origin -u server
'''
import dash_labs as dl
import json
import dash
from dash import ALL, callback_context
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash_extensions.enrich import Dash, Output, Input, State, ServersideOutput
import tifffile as tfi
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from PIL import Image
import plotly.express as px
import pathlib
import base64
import pandas as pd
import io
from io import BytesIO
import re
from utils.controls import controls, controls_nuc, controls_cyto, upload_parm
from utils.Dash_functions import parse_contents
from utils import AIPS_functions as af
from utils import AIPS_module as ai
import pathlib

UPLOAD_DIRECTORY = "/app_uploaded_files"

# style_nav = {'background-color':'#01183A',
#           'max-width': '550px',
#           'width': '100%'}

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.JOURNAL, dbc.icons.FONT_AWESOME],
    prevent_initial_callbacks=True,
    suppress_callback_exceptions=True
)

nav_bar = dbc.Nav(
    [
    dbc.NavItem(dbc.NavLink('display',id='dis', href=dash.page_registry['pages.Image_display']['path'], active=True,disabled=False,class_name='page-link')),
    html.Div(className='separator'),
    dbc.NavItem(dbc.NavLink('Nucleus',id='nuc', href=dash.page_registry['pages.Nucleus_segmentation']['path'], active=False,disabled=True,class_name='page-link')),
    html.Div(className='separator'),
    dbc.NavItem(dbc.NavLink('Target',id='tar', href=dash.page_registry['pages.target_segmentation']['path'], active=False,disabled=True,class_name='page-link')),
    html.Div(className='separator'),
    dbc.NavItem(dbc.NavLink('Download parameters',id='down', href=dash.page_registry['pages.download_parametars']['path'],active=False, disabled=True,class_name='page-link')),
    dbc.NavItem(children=[
                dbc.DropdownMenu(
                        children = [
                        dbc.DropdownMenuItem("Nucleus count predict", href=dash.page_registry['pages.modules.Nuclues_count_predict']['path']),
                        dbc.DropdownMenuItem("SVM target classification", href=dash.page_registry['pages.modules.SVM_target_classification']['path']),
                        # dbc.DropdownMenuItem("Map and save single cell", href=dash.page_registry['pages.modules.Single_cell_download']['path']),
                        ],
                        label="Modules")
                        ]),
                    ],
        pills=True,
        fill=False,
        justified=True,
        navbar=True,
        #className="page-link"
    )

app.layout = dbc.Container(
    [
        html.H1("Optical Pooled Cell Sorting Platform",className='header'),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                 dcc.Upload(
                        id='upload-image',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),className='file_uploader',multiple=True
                    ),
                html.Button('Submit', id='submit-val', n_clicks=0),
                dbc.Accordion(
                            [
                                dbc.AccordionItem(children=
                                [
                        controls,
                                ],title='Image configuration'),
                                dbc.AccordionItem(children=
                                [
                        controls_nuc,
                                ],title='Seed segmentation config'),
                                dbc.AccordionItem(children=
                                [
                        controls_cyto,
                                ], title='Target segmentation config'),
                                dbc.AccordionItem(children=
                                [
                        upload_parm,

                                ], title='Update parameters'),
                            ], start_collapsed=True),
                html.Br(),
                html.Div(id='Tab_table_display'),
            ], width={"size": 4}),
            dbc.Col([
                nav_bar,
                html.Div(id='Tab_slice'),
                dash.page_container,
                html.Div(id='ch_holder', children=[]),
                html.Div(id='ch2_holder',children=[]),
                dcc.Store(id='slice_selc',data=None),
                dcc.Store(id='ch_slice',data=None),
                dcc.Store(id='ch2_slice',data=None),
                dcc.Store(id='json_img_ch',data=None),
                dcc.Store(id='json_img_ch2',data=None),
                dcc.Store(id='json_react', data=None), # rectangle for memory reduction
                dcc.Store(id='offset_store',data=None),
                dcc.Store(id='offset_cyto_store',data=None),
                dcc.Store(id='slider-memory-scale', data=None),
                html.Div(id="test-image-name",hidden=True),
                dcc.Interval(id = 'interval',interval=1000,max_intervals=2,disabled=True)
            ]),
            ])],
    fluid=True)


@app.callback(
    [Output('act_ch', 'value'),
    Output('block_size', 'value'),
    Output('offset_store', 'data'),
    Output('rmv_object_nuc', 'value'),
    Output('block_size_cyto', 'value'),
    Output('offset_cyto_store', 'data'),
    Output('global_ther', 'value'),
    Output('rmv_object_cyto', 'value'),
    Output('rmv_object_cyto_small', 'value'),
    Output('set-val','n_clicks'),
    Output('set-val-cyto','n_clicks'),
     ],
    [Input('submit-parameters', 'n_clicks'),
     State('upload-csv', 'filename'),
     State('upload-csv', 'contents')])
def Load_image(n,pram,cont):
    if n < 1:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,\
               dash.no_update,dash.no_update, dash.no_update, dash.no_update,dash.no_update,dash.no_update
    parameters = parse_contents(cont,pram)
    channel = int(parameters['act_ch'][0])
    bs = parameters['block_size'][0]
    os = parameters['offset'][0]
    ron = parameters['rmv_object_nuc'][0]
    bsc = parameters['block_size_cyto'][0]
    osc = parameters['offset_cyto'][0]
    gt = parameters['global_ther'][0]
    roc = parameters['rmv_object_cyto'][0]
    rocs = parameters['rmv_object_cyto_small'][0]
    #mem = parameters['memory_reduction'][0] #(1,2,3,4)
    set_nuc=1
    set_cyt=1
    return channel,bs,os,ron,bsc,osc,gt,roc,rocs,set_nuc,set_cyt


@app.callback(
    [
    ServersideOutput('json_img_ch', 'data'),
    ServersideOutput('json_img_ch2', 'data')],
    [Input('submit-val', 'n_clicks'),
    State('upload-image', 'filename'),
    State('upload-image', 'contents'),
    Input('act_ch', 'value'),
    Input('json_react','data'),
    Input('ch_slice', 'data'),
    Input('ch2_slice', 'data'),
    State('slice_image_on','on'),
     ],memoize=True)
def Load_image(n,image,cont,channel_sel,react,ch_slice,ch2_slice,slice):
    '''
    react: reactangle from draw compnante of user
    '''
    if n == 0:
        return dash.no_update,dash.no_update
    if slice is True and ch_slice is not None:
        ch_ = ch_slice
        ch2_ = ch2_slice
    else:
        content_string = cont[0].split('data:image/tiff;base64,')[1]
        decoded = base64.b64decode(content_string)
        pixels = tfi.imread(io.BytesIO(decoded))
        pixels_float = pixels.astype('float64')
        img = pixels_float / 65535.000
        if channel_sel == 1:
            ch_ = img[0,:,:]
            ch2_ = img[1,:,:]
        else:
            ch_ = img[1,:,:]
            ch2_ = img[0,:,:]
        if react is not None:
            y0, y1, x0, x1 = react
            ch_ = ch_[y0:y1, x0:x1]
            ch2_ = ch2_[y0:y1, x0:x1]
    json_object_img_ch = ch_
    json_object_img_ch2 = ch2_
    return json_object_img_ch,json_object_img_ch2
################################################
#     Slice image if it is too large
################################################

@app.callback([
    Output('ch_holder', 'children'),
    Output('ch2_holder','children')],
   [
    Input('slice_image_on','on'),
    State('graduated-bar-slice_image','value'),
    State('json_img_ch', 'data'),
    State('json_img_ch2', 'data'),
    State('ch_holder', 'children'),
    State('ch2_holder', 'children'),
   ])
def store_slice(slice,slice_size,ch,ch2,ch_child,ch2_child):
    if slice == False:
        return dash.no_update,dash.no_update
    else:
        ch = np.array(ch)
        ch2 = np.array(ch2)
        H = np.shape(ch2)[0] // slice_size
        W = np.shape(ch2)[1] // slice_size
        tiles_ch = [ch[x:x + H, y:y + W] for x in range(0, ch.shape[0], H) for y in range(0, ch.shape[1], W)]
        tiles_ch2 = [ch2[x:x + H, y:y + W] for x in range(0, ch2.shape[0], H) for y in range(0, ch2.shape[1], W)]
        count = 0
        for t_ch,t_ch2 in zip(tiles_ch,tiles_ch2):
            count += 1
            new_store_ch = dcc.Store(id={'type': 'store_obj_ch',
                                      'index': count},
                                  data=tiles_ch)
            new_store_ch2 = dcc.Store(id={'type': 'store_obj_ch2',
                                      'index': count},
                                  data=tiles_ch2)
            ch_child.append(new_store_ch)
            ch2_child.append(new_store_ch2)
        return ch_child,ch2_child
#
# display the tabs for slice selection
@app.callback(
    Output('Tab_slice', 'children'),
    Input({'type': 'store_obj_ch', 'index': ALL}, 'data'))
def display_tab(data):
    count = np.linspace(1,len(data),len(data))
    return [html.Div(children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Slice number: {}".format(int(c)),
                                   id ={'type':'Image_number_slice',
                                        'index':int(c)}) for c in count])
                             ])
                        ])
                    ]

# # save slice
@app.callback([
    ServersideOutput('ch_slice', 'data'),
    ServersideOutput('ch2_slice', 'data'),
    Output('slice_selc','data')],
    [Input({'type': 'store_obj_ch', 'index': ALL}, 'data'),
    Input({'type': 'store_obj_ch2', 'index': ALL}, 'data'),
    Input({'type':'Image_number_slice','index':ALL}, 'n_clicks')])
def save_img_slice(ch_slice,ch2_slice,slice_click):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    active_index = re.sub(',.*', '', changed_id).split(':')[1]
    ch_ = np.array(ch_slice[int(active_index) - 1])[int(active_index) - 1, :, :]
    ch2_ = np.array(ch2_slice[int(active_index) - 1])[int(active_index) - 1, :, :]
    return ch_, ch2_,active_index



@app.callback(Output('graduated-bar', 'value'),
              Input('graduated-bar-slider', 'value'))
def update_bar(bar_slider):
    return bar_slider

@app.callback(Output('graduated-bar-nuc-zoom', 'value'),
              Input('graduated-bar-slider-nuc-zoom', 'value'))
def update_bar2(bar_slider_zoom):
    return bar_slider_zoom


@app.callback(
    [Output('offset', 'min'),
     Output('offset', 'max'),
     Output('offset', 'marks'),
     Output('offset', 'value'),
     Output('offset', 'step')],
    [Input('set-val','n_clicks'),
    Input('graduated-bar-slider-nuc-zoom', 'value'),
    State('json_img_ch', 'data'),
     State('Auto-nuc', 'on'),
     State('offset', 'value'),
     State('graduated-bar-slider', 'value'),
     State('upload-image', 'filename'),
     State('act_ch', 'value'),
     State('submit-parameters', 'n_clicks'),
     State('upload-csv', 'filename'),
     State('upload-csv', 'contents'),
     ])
def Updat_offset(set_n,bar_zoom,ch,au,offset_input,bar_ind,image_input,channel_sel,n_parm,pram,cont):
    if au:
        memory_index = {1: [0.25, 4], 2: [0.125, 8], 3: [0.062516, 16], 4: [0.031258, 32]}
        ch_ = np.array(ch)
        med_nuc = np.median(ch_) / 400
        norm = np.random.normal(med_nuc, 0.001*bar_ind, 100)
        offset_pred = []
        for norm_ in norm:
            AIPS_object = ai.Segment_over_seed(Image_name=image_input[0], path=UPLOAD_DIRECTORY, rmv_object_nuc=0.9,
                                               block_size=59, offset=norm_,
                                               block_size_cyto=9, offset_cyto=0.0004, global_ther=0.4,
                                               rmv_object_cyto=0.99, rmv_object_cyto_small=0.9, remove_border=True)
            nuc_s = AIPS_object.Nucleus_segmentation(ch_, inv=False,rescale_image=True,scale_factor=memory_index[1])
            offset_pred = norm_
            len_table = len(nuc_s['tabale_init'])
            if len_table > 3:
                break
        norm = np.random.normal(offset_pred, 0.001, 100)
        min_val = round(np.min(norm), 4)
        max_val = round(np.max(norm), 4)
        steps = (np.max(norm) - np.min(norm)) / bar_zoom
        value_marks = {i: i for i in [min_val, max_val]}
        return [min_val, max_val, value_marks, offset_pred,steps]
    else:
        if n_parm > 0:
            parameters = parse_contents(cont, pram)
            os = parameters['offset'][0]
            norm = np.random.normal(os, 0.001, 100)
            min_val = round(np.min(norm), 4)
            max_val = round(np.max(norm), 4)
            steps = (np.max(os) - np.min(os)) / bar_zoom
            value_marks = {i: i for i in [min_val, max_val]}
            offset_pred = os
        else:
            min_val = 0.001
            max_val = 0.8
            value_marks = {i: i for i in [0.001, 0.8]}
            offset_pred = offset_input
            steps = 0.001
        return [min_val, max_val, value_marks, offset_pred,steps]

'''
    Initiate offset parameter prediction for Cytosol segmentation
'''

@app.callback(Output('graduated-bar-cyto', 'value'),
              Input('graduated-bar-slider-cyto', 'value'))
def update_bar_cyto(bar_slider_cyto):
    return bar_slider_cyto

@app.callback(Output('graduated-bar-cyto-zoom', 'value'),
              Input('graduated-bar-slider-cyto-zoom', 'value'))
def update_bar3(bar_slider_cyto_zoom):
    return bar_slider_cyto_zoom


@app.callback(
    [Output('offset_cyto', 'min'),
     Output('offset_cyto', 'max'),
     Output('offset_cyto', 'marks'),
     Output('offset_cyto', 'value'),
     Output('offset_cyto', 'step')],
    [Input('set-val-cyto','n_clicks'),
     Input('graduated-bar-slider-cyto-zoom', 'value'),
    State('json_img_ch2', 'data'),
    State('Auto-cyto', 'on'),
    State('offset_cyto', 'value'),
    State('graduated-bar-cyto', 'value'),
    State('upload-image', 'filename'),
    State('act_ch', 'value'),
    State('submit-parameters', 'n_clicks'),
    State('upload-csv', 'filename'),
    State('upload-csv', 'contents'),
     ])
def Updat_offset_cyto(set_n,bar_zoom_cyto,ch2,au,offset_input,bar_ind,image_input,channel_sel,n_parm,pram,cont):
    if au:
        memory_index = {1: [0.25, 4], 2: [0.125, 8], 3: [0.062516, 16], 4: [0.031258, 32]}
        ch2_ = np.array(ch2)
        med_nuc = np.median(ch2_) / 400
        norm = np.random.normal(med_nuc, 0.001*bar_ind, 100)
        offset_pred = []
        for norm_ in norm:
            AIPS_object = ai.Segment_over_seed(Image_name=image_input[0], path=UPLOAD_DIRECTORY, rmv_object_nuc=0.9,
                                               block_size=59, offset=norm_,
                                               block_size_cyto=9, offset_cyto=0.0004, global_ther=0.4,
                                               rmv_object_cyto=0.99, rmv_object_cyto_small=0.9, remove_border=True)
            nuc_s = AIPS_object.Nucleus_segmentation(ch2_, inv=False,rescale_image=True,scale_factor=memory_index[1])
            offset_pred = norm_
            len_table = len(nuc_s['tabale_init'])
            if len_table > 3:
                break
        norm = np.random.normal(offset_pred, 0.001, 100)
        min_val = round(np.min(norm), 4)
        max_val = round(np.max(norm), 4)
        steps = (np.max(norm) - np.min(norm)) / bar_zoom_cyto
        value_marks = {i: i for i in [min_val, max_val]}
        return [min_val, max_val, value_marks, offset_pred,steps]
    else:
        if n_parm > 0:
            parameters = parse_contents(cont, pram)
            osc = parameters['offset_cyto'][0]
            norm = np.random.normal(osc, 0.001, 100)
            min_val = round(np.min(norm), 4)
            max_val = round(np.max(norm), 4)
            steps = (np.max(os) - np.min(os)) / bar_zoom_cyto
            value_marks = {i: i for i in [min_val, max_val]}
            offset_pred = os
        else:
            min_val = 0.001
            max_val = 0.8
            value_marks = {i: i for i in [0.001, 0.8]}
            offset_pred = offset_input
            steps = 0.001
        return [min_val, max_val, value_marks, offset_pred,steps]

if __name__ == "__main__":
    #app.run_server(debug=False)
    app.run_server(debug=True, use_reloader=False)
    #app.run_server()
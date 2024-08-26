import dash
dash.register_page(__name__,name ='Image_display' ,path='/')
from dash_extensions.enrich import Dash, Output, Input, State, ServersideOutput
from dash import  html, dcc, callback
import dash_bootstrap_components as dbc
import dash_daq as daq
import json
from utils import AIPS_functions as af
from utils import AIPS_module as ai
import numpy as np
from PIL import Image
import plotly.express as px

UPLOAD_DIRECTORY = "/app_uploaded_files"

layout = html.Div([
    dbc.Alert(id='alert_display', is_open=False),
    dcc.Loading(html.Div(id='img-output'), type="circle", style={'height': '100%', 'width': '100%'}),
    ])
@callback(
    [Output('img-output', 'children'),
     Output('alert_display', 'is_open'),
     Output('nuc', 'active'),
     Output('nuc', 'disabled'),],
    [Input('json_img_ch', 'data'),
    Input('json_img_ch2', 'data'),
    State('upload-image', 'filename'),
    State('upload-image', 'contents'),
    Input('act_ch', 'value'),
    State('Auto-nuc', 'value'),
    Input('high_pass', 'value'),
    Input('low_pass', 'value'),
    Input('block_size','value'),
    Input('offset','value'),
    Input('rmv_object_nuc','value'),
    Input('block_size_cyto', 'value'),
    State('Auto-cyto', 'value'),
    Input('offset_cyto', 'value'),
    Input('global_ther', 'value'),
    Input('rmv_object_cyto', 'value'),
    Input('rmv_object_cyto_small', 'value'),
     ],
    suppress_callback_exceptions=True)
def Parameters_initiation(ch,ch2, image,cont,channel,int_on_nuc,high,low,bs,os,ron,bsc,int_on_cyto,osc,gt,roc,rocs):
    memory_index =  {1:[0.25,4],2:[0.125,8],3:[0.062516,16],4:[0.031258,32]}
    AIPS_object = ai.Segment_over_seed(Image_name=image[0], path=UPLOAD_DIRECTORY,rmv_object_nuc=ron,
                                       block_size=bs,
                                       offset=os,
                                       block_size_cyto=bsc, offset_cyto=osc, global_ther=gt, rmv_object_cyto=roc,
                                       rmv_object_cyto_small=rocs, remove_border=False)
    ch_ = np.array(ch)
    ch2_ = np.array(ch2)
    if np.shape(ch_)[0] > 512:
        alert_massage = True
    else:
        alert_massage = False
    ch_3c = af.gray_scale_3ch(ch_)
    ch2_3c = af.gray_scale_3ch(ch2_)
    nuc_s = AIPS_object.Nucleus_segmentation(ch_,rescale_image=True,scale_factor=memory_index[1])
    #seg = AIPS_object.Cytosol_segmentation(ch_, ch2_, nuc_s['sort_mask'], nuc_s['sort_mask_bin'], rescale_image=True)
    # dict_ = {'img':img,'nuc':nuc_s,'seg':seg}
    # try to work on img
    nmask2 = nuc_s['nmask2']
    nmask4 = nuc_s['nmask4']
    sort_mask = nuc_s['sort_mask']
    sort_mask_bin = nuc_s['sort_mask_bin']
    sort_mask_bin = np.array(sort_mask_bin, dtype=np.int8)
    table = nuc_s['table']
    seg = AIPS_object.Cytosol_segmentation(ch_, ch2_, sort_mask, sort_mask_bin, rescale_image=True,scale_factor=memory_index[1])
    cell_mask_1 = seg['cell_mask_1']
    combine = seg['combine']
    cseg_mask = seg['cseg_mask']
    cseg_mask = seg['cseg_mask']
    info_table = seg['info_table']
    mask_unfiltered = seg['mask_unfiltered']
    table_unfiltered = seg['table_unfiltered']
    image_size_x = np.shape(ch_)[1]
    image_size_y = np.shape(ch2_)[1]
    try:
        med_seed = int(np.median(table['area']))
        len_seed = len(table)
    except:
        med_seed = 'None'
        len_seed = 'None'
    try:
        med_cyto = int(np.median(info_table['area']))
        len_cyto = len(info_table)
    except:
        med_cyto = 'None'
        len_cyto = 'None'
    '''
    Display image
    '''
    if channel == 1:
        Channel_number_1 = 'Channel 1'
        Channel_number_2 = 'Channel 2'
    else:
        Channel_number_1 = 'Channel 2'
        Channel_number_2 = 'Channel 1'
    pix = af.show_image_adjust(ch_, low_prec=low, up_prec=high)
    pix = pix * 65535.000
    im_pil = Image.fromarray(np.uint16(pix))
    fig_ch = px.imshow(im_pil, binary_string=True, binary_backend="jpg",width=500,height=500,title='Seed:'+ Channel_number_1,binary_compression_level=9).update_xaxes(showticklabels = False).update_yaxes(showticklabels = False)
    fig_ch.update_layout(title_x=0.5,dragmode="drawrect")
    pix_2 = af.show_image_adjust(ch2_, low_prec=low, up_prec=high)
    pix_2 = pix_2 * 65535.000
    im_pil = Image.fromarray(np.uint16(pix_2))
    fig_ch2 = px.imshow(im_pil, binary_string=True, binary_backend="jpg",width=500,height=500,title='Target:'+ Channel_number_2,binary_compression_level=9).update_xaxes(showticklabels = False).update_yaxes(showticklabels = False)
    fig_ch2.update_layout(title_x=0.5,dragmode="drawrect")
    return [
            dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            id="graph_ch",
                            figure=fig_ch), md=6),
                    dbc.Col(
                        dcc.Graph(
                            id="graph_ch2",
                            figure=fig_ch2), md=6),
                    ]),
            html.Br(),
            html.Br(),
            dbc.Col([
                dbc.Row(html.Label('Image parameters:')),
            html.P([
                dbc.Row(html.Label("Image size: {} x {}".format(image_size_x, image_size_y))),
                html.Br(),
                dbc.Row(html.Label('Seed Image parameters:')),
                html.P([
                    dbc.Row(html.Label("Median object area: {}".format(med_seed))),
                    dbc.Row(html.Label("Number of objects detected: {}".format(len_seed))), ]),
                dbc.Row(html.Label('Target Image Parameters :')),
                html.P([
                    dbc.Row(html.Label("Median object area: {}".format(med_cyto))),
                    dbc.Row(html.Label("Number of objects detected: {}".format(len_cyto)))]),
            ])]),
               daq.BooleanSwitch(
                   label='Clear selections',
                   id='clear_selc',
                   disabled=False,
                   on=False,
               )
            ],alert_massage,True,False


# if loading is slow than reduce image size
@callback([
    Output("json_react", "data"),
    Output("json_react", "clear_data")],
    [Input("graph_ch", "relayoutData"),
    Input("graph_ch2", "relayoutData"),
    Input("clear_selc", "on"),])
def on_new_annotation(relayout_data_ch,relayout_data_ch2,clear):
    if relayout_data_ch is not None or relayout_data_ch2 is not None:
        if relayout_data_ch is None:
            relayout_data = relayout_data_ch2
        else:
            relayout_data = relayout_data_ch
        last_shape = relayout_data["shapes"][-1]
        # shape coordinates are floats, we need to convert to ints for slicing
        x0, y0 = int(last_shape["x0"]), int(last_shape["y0"])
        x1, y1 = int(last_shape["x1"]), int(last_shape["y1"])
        size = [y0,y1,x0,x1]
        size_data = size
        clear_json = False
        return size_data, clear_json
    elif clear is True:
        size_data = None
        clear_json = True
        return size_data, clear_json
    else:
        return dash.no_update,dash.no_update



import dash
dash.register_page(__name__, name='target_segmentation', path='/target_segmentation')
from dash_extensions.enrich import Dash, Output, Input, State, ServersideOutput
from dash import  html, dcc, callback
import dash_bootstrap_components as dbc
import json
from utils import AIPS_functions as af
from utils import AIPS_module as ai
import numpy as np
from PIL import Image
import plotly.express as px

UPLOAD_DIRECTORY = "/app_uploaded_files"

layout = html.Div([
    dcc.Loading(html.Div(id='img-cell-output'), type="circle", style={'height': '100%', 'width': '100%'})
    ])


@callback(
    [Output('img-cell-output', 'children'),
    Output('down', 'active'),
    Output('down', 'disabled')],
    [
    Input('json_img_ch', 'data'),
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
    Input('rmv_object_cyto_small', 'value')]
    ,suppress_callback_exceptions=True)
def Parameters_initiation(ch,ch2, image,cont,channel,int_on_nuc,high,low,bs,os,ron,bsc,int_on_cyto,osc,gt,roc,rocs):
    memory_index =  {1:[0.25,4],2:[0.125,8],3:[0.062516,16],4:[0.031258,32]}
    AIPS_object = ai.Segment_over_seed(Image_name=image[0], path=UPLOAD_DIRECTORY,rmv_object_nuc=ron,
                                       block_size=bs,
                                       offset=os,
                                       block_size_cyto=bsc, offset_cyto=osc, global_ther=gt, rmv_object_cyto=roc,
                                       rmv_object_cyto_small=rocs, remove_border=False)
    ch_ = np.array(ch)
    ch2_ = np.array(ch2)
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
    fig_im_pil_sort_mask = af.plot_composite_image(ch2_3c, sort_mask, fig_title='RGB map - seed', alpha=0.2)
    fig_im_pil_cell_mask_2 = af.px_pil_figure(combine, bit=1, mask_name='_cell_mask',
                                              fig_title='Local threshold map - seed', wh=500)
    fig_im_pil_mask_unfiltered = af.plot_composite_image(ch2_3c, mask_unfiltered, fig_title='Mask - Target', alpha=0.2)
    # binary_to_ROI (slower)
    cseg_mask = np.array(cseg_mask, dtype=np.int32)
    ch2_cseg_mask = af.rgb_file_gray_scale(ch2_, mask=cseg_mask, channel=0)
    fig_im_pil_cseg_mask = af.px_pil_figure(ch2_cseg_mask, bit=3, mask_name='_cseg',
                                            fig_title='Mask - Target (filterd)', wh=500)

    len_unfiltered_table = table_unfiltered
    return [
        dbc.Row([
            dbc.Col(
                dcc.Graph(
                    id="graph_fig_im_pil_sort_mask",
                    figure=fig_im_pil_sort_mask), md=6),
            dbc.Col(
                dcc.Graph(
                    id="graph_local_thershold",
                    figure=fig_im_pil_cell_mask_2), md=6),
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Graph(
                    id="Mask_Target_unfiltered",
                    figure=fig_im_pil_mask_unfiltered), md=6),
            dbc.Col(
                dcc.Graph(
                    id="Mask_Target",
                    figure=fig_im_pil_cseg_mask), md=6),
        ]),
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Accordion([
                dbc.AccordionItem(
                    title="Cytosole segmentation inspection", children=[
                        dbc.Row(html.Label(
                            "Number of objects detected before filtering: {}".format(len_unfiltered_table.iloc[0, 0]))),
                        dbc.Row(html.Label(
                            "Number of objects - large filtered: {}".format(len_unfiltered_table.iloc[0, 1]))),
                        dbc.Row(html.Label(
                            "Number of objects - Small filtered:: {}".format(len_unfiltered_table.iloc[0, 2])))
                    ])
            ])])
             ],True,False
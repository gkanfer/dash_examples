import time
import skimage.transform
import tifffile as tfi
import numpy as np
integer = np.linspace(1,99,10)
for i in integer:
    print(int(i))
list(np.around(np.arange(1, 99, 10),1))

integer_1_99 = list(np.around(np.arange(1, 99, 10),1)-1)
integer_1_99[0] = 1
integer_1_99.append(99)
marks = {integer_1_99[i]: '{}'.format(integer_1_99[i]) for i in range(len(integer_1_99))}

from skimage.filters import threshold_local
from scipy.ndimage.morphology import binary_opening
import skimage.morphology as sm
from skimage.segmentation import watershed
from skimage import measure
import os
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import dash_daq as daq
import json
import dash
import dash.exceptions
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output, State
import tifffile as tfi
import glob
import os
import numpy as np
from skimage.exposure import rescale_intensity, histogram
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage.morphology import binary_opening, binary_erosion, binary_dilation
from PIL import Image, ImageEnhance
import base64
import pandas as pd
import re
from random import randint
from io import BytesIO
from flask_caching import Cache
from dash.long_callback import DiskcacheLongCallbackManager
import plotly.express as px
from skimage import io, filters, measure, color, img_as_ubyte

from utils.controls import controls, controls_nuc, controls_cyto
from utils import AIPS_functions as af
from utils import AIPS_module as ai
from utils import display_and_xml as dx

import pathlib
from app import app
from utils.controls import controls, controls_nuc, controls_cyto
from utils import AIPS_functions as af
from utils import AIPS_module as ai
from utils import display_and_xml as dx
from utils.Display_composit import image_with_contour, countor_map
import re
set = '{"index":1,"type":"Image_number_slice"}.n_clicks'
text = re.search('"index":.',set)
x = re.sub(',.*','', set).split(':')[1]
x.split(':')[1]

Image.open(BytesIO(base64.b64decode(image_string[22:])))
path = '/Users/kanferg/Desktop/NIH_Youle/Colobration/Elliot/drive-download-20220216T203417Z-001/'
#tifWT_37C_DMSO.tif
AIPS_object = ai.Segment_over_seed(Image_name='tifWT_37C_DMSO.tif', path=path, rmv_object_nuc=0.5, block_size=83,
                                           offset=0.00001,block_size_cyto=17, offset_cyto=-0.00004, global_ther=0.4, rmv_object_cyto=0.7,
                                           rmv_object_cyto_small=0.1, remove_border=True)

path = '/Users/kanferg/Desktop/NIH_Youle/Python_projacts_general/dash/AIPS_Dash_Final/app_uploaded_files/'
#Composite.tif10.tif
AIPS_object = ai.Segment_over_seed(Image_name='dmsot0273_0003-512.tif', path=path, rmv_object_nuc=0.5, block_size=83,
                                           offset=0.00001,block_size_cyto=17, offset_cyto=-0.004, global_ther=0.4, rmv_object_cyto=0.99,
                                           rmv_object_cyto_small=0.4, remove_border=True)
img = AIPS_object.load_image()
memory_index = {1: [0.25, 4], 2: [0.125, 8], 3: [0.062516, 16], 4: [0.031258, 32]}
ch = img['1']
ch2 = img['0']
nuc_s = AIPS_object.Nucleus_segmentation(img['1'], inv=False, for_dash=False,rescale_image=False,scale_factor=memory_index[1])
seg = AIPS_object.Cytosol_segmentation(ch, ch2, nuc_s['sort_mask'], nuc_s['sort_mask_bin'], rescale_image=False,scale_factor=memory_index[1])

H = np.shape(ch2)[0] // 2
W = np.shape(ch2)[1] // 2
tiles_ch2 = [ch2[x:x + H, y:y + W] for x in range(0, ch2.shape[0], H) for y in range(0, ch2.shape[1], W)]
tiles_ch = [ch[x:x + H, y:y + W] for x in range(0, ch.shape[0], H) for y in range(0, ch.shape[1], W)]
for t_ch,t_ch2 in zip(tiles_ch,tiles_ch2):
    print(np.shape(t_ch2))

    # count += 1
    # new_store = dcc.Store(id={'type': 'store_obj',
    #                           'index': count},
    #                       data=tile)


nmask2 = nuc_s['nmask2']
nmask4 = nuc_s['nmask4']
sort_mask = nuc_s['sort_mask']
plt.imshow(sort_mask)
sort_mask_bin = nuc_s['sort_mask_bin']
sort_mask_bin = np.array(sort_mask_bin, dtype=np.int8)
table = nuc_s['table']
cell_mask_1 = seg['cell_mask_1']
combine = seg['combine']
cseg_mask = seg['cseg_mask']
plt.imshow(cseg_mask)


prop_names = [
        "label",
        "area",
        "eccentricity",
        "centroid_weighted",
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
    cseg_mask, intensity_image=ch2, properties=prop_names
)

table_center = pd.DataFrame(measure.regionprops_table(cseg_mask,
                            intensity_image= ch2,
                            properties=['label', 'centroid_weighted',])
                         ).set_index('label')

H = np.shape(ch)[0] // 2
W = np.shape(ch)[1] // 2
tiles = [ch[x:x + H, y:y + W] for x in range(0, ch.shape[0], H) for y in range(0, ch.shape[1], W)]
print(len(tiles))

fig, ax = plt.subplots(2, 3, figsize=(15, 15))
ax[0][0].imshow(tiles[0])
ax[0][1].imshow(tiles[1])
ax[0][2].imshow(tiles[2])
ax[1][0].imshow(tiles[3])
ax[1][1].imshow(ch)
ax[1][2].imshow(ch2)
plt.imshow(np.array(tiles[0]))
plt.imshow(np.array(tiles[0]))




info_table = seg['info_table']
mask_unfiltered = seg['mask_unfiltered']
table_unfiltered = seg['table_unfiltered']
image_size_x = np.shape(ch)[1]
image_size_y = np.shape(ch2)[1]

ch_3c = af.gray_scale_3ch(ch)
fig_im_pil_sort_mask = af.plot_composite_image(ch_3c, sort_mask, fig_title='RGB map - seed', alpha=0.2)
#fig_im_pil_sort_mask.show()

plt.imshow(cseg_mask)
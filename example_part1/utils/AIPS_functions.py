'''
Function for AIPS DASH
'''
import xml.etree.ElementTree as xml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# from PIL import fromarray
import plotly.express as px
from skimage.filters import threshold_local
from scipy.ndimage.morphology import binary_opening
from skimage import io, filters, measure, color, img_as_ubyte
import skimage.morphology as sm
from skimage.segmentation import watershed
from skimage import measure
from skimage.exposure import rescale_intensity
import os
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import base64
from datetime import datetime
import re
from utils.display_and_xml import unique_rand
from utils import display_and_xml as dx

def segmentation_2ch(ch,ch2, rmv_object_nuc, block_size, offset,int_nuc, cyto_seg,
                     block_size_cyto, int_cyto ,offset_cyto, global_ther, rmv_object_cyto, rmv_object_cyto_small):
    '''
       Function for exploring the parameters for simple threshold based segmentation
       Prefer Nucleus segmentation
       Args:
           ch: Input image (tifffile image object)
           block_size: Detect local edges 1-99 odd
           offset: Detect local edges 0.001-0.9 odd
           rmv_object_nuc: percentile of cells to remove, 0.01-0.99
           cyto_seg: 1 or 0
           block_size_cyto: Detect local edges 1-99 odd
           offset_cyto: Detect local edges 0.001-0.9 odd
           global_ther: Percentile
           rmv_object_cyto:  percentile of cells to remove, 0.01-0.99
           rmv_object_cyto_small:  percentile of cells to remove, 0.01-0.99
       Returns:
            nmask2: local threshold binary map (eg nucleus)
            nmask4: local threshold binary map post opening (eg nucleus)
            sort_mask: RGB segmented image output first channel for mask (eg nucleus)
            cell_mask_2: local threshold binary map (eg cytoplasm)
            combine: global threshold binary map (eg cytoplasm)
            cseg_mask: RGB segmented image output first channel for mask (eg nucleus)
            test: Area table seed
            test2: Area table cytosol
    '''
    if int_nuc[0]==1:
        nmask = threshold_local(ch, block_size, "mean", np.median(np.ravel(ch))/10)
    else:
        nmask = threshold_local(ch, block_size, "mean", offset)
    nmask2 = ch > nmask
    nmask3 = binary_opening(nmask2, structure=np.ones((3, 3))).astype(np.float64)
    nmask4 = binary_fill_holes(nmask3)
    label_objects = sm.label(nmask4, background=0)
    info_table = pd.DataFrame(
        measure.regionprops_table(
            label_objects,
            intensity_image=ch,
            properties=['area', 'label','coords'],
        )).set_index('label')
    #info_table.hist(column='area', bins=100)
    test = info_table[info_table['area'] < info_table['area'].quantile(q=rmv_object_nuc)]
    sort_mask = label_objects
    if len(test) > 0:
        x = np.concatenate(np.array(test['coords']))
        sort_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
    else:
        test = info_table[info_table['area'] < info_table['area'].quantile(q=0.5)]
        x = np.concatenate(np.array(test['coords']))
        sort_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
    sort_mask_bin = np.where(sort_mask > 0, 1, 0)
    if cyto_seg[0]==1:
        if int_cyto[0]==1:
            ther_cell = threshold_local(ch2, block_size_cyto, "gaussian", np.median(np.ravel(ch2))/10)
        else:
            ther_cell = threshold_local(ch2, block_size_cyto, "gaussian", offset_cyto)
        cell_mask_1 = ch2 > ther_cell
        cell_mask_2 = binary_opening(cell_mask_1, structure=np.ones((3, 3))).astype(np.float64)
        quntile_num = np.quantile(ch2, global_ther)
        cell_mask_3 = np.where(ch2 > quntile_num, 1, 0)
        combine = cell_mask_2
        combine[cell_mask_3 > combine] = cell_mask_3[cell_mask_3 > combine]
        combine[sort_mask_bin > combine] = sort_mask_bin[sort_mask_bin > combine]
        cseg = watershed(np.ones_like(sort_mask_bin), sort_mask, mask=cell_mask_2)
        csegg = watershed(np.ones_like(sort_mask), cseg, mask=combine)
        info_table = pd.DataFrame(
            measure.regionprops_table(
                csegg,
                intensity_image=ch2,
                properties=['area', 'label','coords'],
            )).set_index('label')
       #info_table.hist(column='area', bins=100)
        ############# remove large object ################
        cseg_mask = csegg
        test1 = info_table[info_table['area'] > info_table['area'].quantile(q=rmv_object_cyto)]
        if len(test1) > 0:
            x = np.concatenate(np.array(test1['coords']))
            cseg_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
        else:
            test1 = info_table[info_table['area'] > info_table['area'].quantile(q=0.5)]
            x = np.concatenate(np.array(test1['coords']))
            cseg_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
        ############# remove small object ################
        test2 = info_table[info_table['area'] < info_table['area'].quantile(q=rmv_object_cyto_small)]
        if len(test2) > 0:
            x = np.concatenate(np.array(test2['coords']))
            cseg_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
        else:
            test2 = info_table[info_table['area'] > info_table['area'].quantile(q=0.5)]
            x = np.concatenate(np.array(test2['coords']))
            cseg_mask[tuple(x.T)[0], tuple(x.T)[1]] = 0
        dict = {'nmask2':nmask2, 'nmask4':nmask4, 'sort_mask':sort_mask,'cell_mask_1':cell_mask_1,
                'combine':cell_mask_3, 'cseg_mask':cseg_mask,'test':test, 'test2':test2}
        return dict
    else:
        dict = {'nmask2': nmask2, 'nmask4': nmask4,'test':test ,'sort_mask': sort_mask}
        return dict



def show_image_adjust(image, low_prec, up_prec):
    """
    image= np array 2d
    low/up precentile border of the image
    """
    percentiles = np.percentile(image, (low_prec, up_prec))
    scaled_ch1 = rescale_intensity(image, in_range=tuple(percentiles))
    return scaled_ch1
    # PIL_scaled_ch1 = Image.fromarray(np.uint16(scaled_ch1))
    # PIL_scaled_ch1.show()
    # return PIL_scaled_ch1

def px_pil_figure(img,bit,mask_name,fig_title,wh):
    '''
    :param img: image input - 3 channel 8 bit image
             bit:1 np.unit16 or 2 np.unit8
             fig_title: title for display on dash
             wh: width and hight in pixels
    :return: encoded_image (e_img)
    '''
    bit = str(img.dtype)
    if bit == "bool":
        # binary
        im_pil = Image.fromarray(img)
    elif bit == "int64":
        # 16 image (normal image)
        im_pil = Image.fromarray(np.uint16(img))
    else:
        # ROI mask
        img_gs = img_as_ubyte(img)
        im_pil = Image.fromarray(img_gs)
    fig_ch = px.imshow(im_pil, binary_string=True, binary_backend="jpg", width=500, height=500,title=fig_title,binary_compression_level=9).update_xaxes(showticklabels=False).update_yaxes(showticklabels = False)
    fig_ch.update_layout(title_x=0.5)
    return fig_ch


def XML_creat(filename,block_size,offset,rmv_object_nuc,block_size_cyto,offset_cyto,global_ther,rmv_object_cyto,rmv_object_cyto_small):
    root = xml.Element("Segment")
    cl = xml.Element("segment") #chiled
    root.append(cl)
    block_size_ = xml.SubElement(cl,"block_size")
    block_size_.text = block_size
    offset_ = xml.SubElement(cl,"offset")
    offset_.text = "13"
    rmv_object_nuc_ = xml.SubElement(cl, "rmv_object_nuc")
    rmv_object_nuc_.text = "rmv_object_nuc"
    block_size_cyto_ = xml.SubElement(cl, "block_size_cyto")
    block_size_cyto_.text = "block_size_cyto"
    offset_cyto_ = xml.SubElement(cl, "offset_cyto")
    offset_cyto_.text = "offset_cyto"
    global_ther_ = xml.SubElement(cl, "global_ther")
    global_ther_.text = "global_ther"
    rmv_object_cyto_ = xml.SubElement(cl, "rmv_object_cyto")
    rmv_object_cyto_.text = "rmv_object_cyto"
    rmv_object_cyto_small_ = xml.SubElement(cl, "rmv_object_cyto_small")
    rmv_object_cyto_small_.text = "rmv_object_cyto_small"
    tree = xml.ElementTree(root)
    with open(filename,'wb') as f:
        tree.write(f)

def seq(start, end, by=None, length_out=None):
    len_provided = True if (length_out is not None) else False
    by_provided = True if (by is not None) else False
    if (not by_provided) & (not len_provided):
        raise ValueError('At least by or n_points must be provided')
    width = end - start
    eps = pow(10.0, -14)
    if by_provided:
        if (abs(by) < eps):
            raise ValueError('by must be non-zero.')
        # Switch direction in case in start and end seems to have been switched (use sign of by to decide this behaviour)
        if start > end and by > 0:
            e = start
            start = end
            end = e
        elif start < end and by < 0:
            e = end
            end = start
            start = e
        absby = abs(by)
        if absby - width < eps:
            length_out = int(width / absby)
        else:
            # by is too great, we assume by is actually length_out
            length_out = int(by)
            by = width / (by - 1)
    else:
        length_out = int(length_out)
        by = width / (length_out - 1)
    out = [float(start)] * length_out
    for i in range(1, length_out):
        out[i] += by * i
    if abs(start + by * length_out - end) < eps:
        out.append(end)
    return out

def rgb_file_gray_scale(input_gs_image,mask=None,channel=None):
    ''' create a 3 channel rgb image from 16bit input image
        optional bin countor image from ROI image
        :parameter
        input_gs_image: 16bit nparray
        mask: 32int roi image
        channel: 0,1,2 (rgb)
        :return
        3 channel stack file 8bit image
    '''
    input_gs_image = (input_gs_image / input_gs_image.max()) * 255
    ch2_u8 = np.uint8(input_gs_image)
    rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
    rgb_input_img[:, :, 0] = ch2_u8
    rgb_input_img[:, :, 1] = ch2_u8
    rgb_input_img[:, :, 2] = ch2_u8
    if mask is not None and len(np.unique(mask)) > 1:
        bin_mask = dx.binary_frame_mask(ch2_u8, mask)
        bin_mask = np.where(bin_mask == 1, True, False)
        if channel is not None:
            rgb_input_img[bin_mask > 0, channel] = 255
        else:
            rgb_input_img[bin_mask > 0, 2] = 255
    return rgb_input_img


def gray_scale_3ch(input_gs_image):
    input_gs_image = (input_gs_image / input_gs_image.max()) * 255
    ch2_u8 = np.uint8(input_gs_image)
    rgb_input_img = np.zeros((np.shape(ch2_u8)[0], np.shape(ch2_u8)[1], 3), dtype=np.uint8)
    rgb_input_img[:, :, 0] = ch2_u8
    rgb_input_img[:, :, 1] = ch2_u8
    rgb_input_img[:, :, 2] = ch2_u8
    return rgb_input_img

def plot_composite_image(img,mask,fig_title,alpha=0.2):
    # apply colors to mask
    '''
    :param img: input 3 channel grayscale image
    :param mask: mask
    :param fig_title: title
    :param alpha: transprancy for blending
    :param img_shape:
    :return:
    '''
    mask = np.array(mask, dtype=np.int32)
    mask_deci = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    cm = plt.get_cmap('CMRmap')
    colored_image = cm(mask_deci)
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
    #RGB pil image
    img_mask = img_as_ubyte(colored_image)
    im_mask_pil = Image.fromarray(img_mask).convert('RGB')
    img_gs = img_as_ubyte(img)
    im_pil = Image.fromarray(img_gs).convert('RGB')
    im3 = Image.blend(im_pil, im_mask_pil, alpha)
    fig_ch = px.imshow(im3, binary_string=True, binary_backend="jpg", width=500, height=500, title=fig_title,
                       binary_compression_level=0).update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig_ch.update_layout(title_x=0.5)
    return fig_ch

def save_pil_to_directory(img,bit,mask_name,output_dir = 'temp',mask = None, merge_mask = None,channel=None):
    '''
    Save image composite with ROI
    :param img: image input
             bit:1 np.unit16 or 2 np.unit8
             merge_mask: ROI - ROI + 3ch greyscale input OR  -BIN -  bin + 3ch greyscale
             channel: for display Bin merge  with rgb input
    :return: encoded_image (e_img)
    '''
    bit = str(img.dtype)
    if merge_mask=='ROI':
        # img  must be 3 channel grayscale
        mask = np.array(mask, dtype=np.int32)
        mask_deci = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        cm = plt.get_cmap('CMRmap')
        colored_image = cm(mask_deci)
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
        # RGB pil image
        img_mask = img_as_ubyte(colored_image)
        im_mask_pil = Image.fromarray(img_mask).convert('RGB')
        img_gs = img_as_ubyte(img)
        im3 = Image.fromarray(img_gs).convert('RGB')
        im_pil = Image.blend(im3, im_mask_pil, 0.2)
    elif merge_mask=='BIN':
        #img  must be 3 channel grayscale
        img_bin = rgb_file_gray_scale(img, mask=mask, channel=channel)
        img_gs = img_as_ubyte(img_bin)
        im_pil = Image.fromarray(img_gs)
    elif merge_mask is None:
        if bit == "bool":
            # binary
            im_pil = Image.fromarray(img)
        elif bit == "int64":
            # 16 image (normal image)
            im_pil = Image.fromarray(np.uint16(img))
        else:
            # ROI mask
            img = np.uint8(img)
            roi_index_uni = np.unique(img)
            roi_index_uni = roi_index_uni[roi_index_uni > 1]
            sort_mask_buffer = np.ones((np.shape(img)[0], np.shape(img)[1], 3), dtype=np.uint8)
            for npun in roi_index_uni:
                for i in range(3):
                    sort_mask_buffer[img == npun, i] = unique_rand(2, 255, 1)[0]
            im_pil = Image.fromarray(sort_mask_buffer, mode='RGB')
    filename1 = datetime.now().strftime("%Y%m%d_%H%M%S" + mask_name)
    im_pil.save(os.path.join(output_dir, filename1 + ".png"), format='png')  # this is for image processing
    e_img = base64.b64encode(open(os.path.join('temp', filename1 + ".png"), 'rb').read())
    return e_img

def remove_gradiant_label_border(mask):
    '''
    mask from rescale return with border gradiant which is for the borders
    :return mask no borders
    '''
    mask_ = np.where(mask > 0, mask, 0)
    mask_intact = np.where(np.mod(mask_, 1) > 0, 0, mask_)
    mask_intact = np.array(mask_intact, np.uint32)
    return mask_intact

def img_split_instractions(img):
    '''
    Returns the instructions of split in dictionary
    :parameter img - np.array shape (2 channels)
    '''
    w = np.shape(img)[0]
    h = np.shape(img)[1]
    if w > 1000 and w < 2000:
        dict = {
            1: [0, 0 + w / 2, 0, 0 + h / 2],
            2: [w / 2, w, h / 2, h]
        }
    elif w > 1000 and w > 2000:
        dict = {
            1: [0, 0 + (w / 4), 0, 0 + h / 4],
            2: [w / 4, w / 2, h / 4, h / 2],
            3: [w / 2, (w * 3) / 2, h / 2, (h * 3) / 2],
            4: [(w * 3) / 2, w, (h * 3) / 2, h],
        }
    else:
        dict = None
    return dict

def set_slice(img,num):
    '''
        slice image according to image size
        num
    '''
    w = np.shape(img)[0]
    h = np.shape(img)[1]
    dict = {
        1: [0, 0 + (w / 4), 0, 0 + h / 4],
        2: [w / 4, w / 2, h / 4, h / 2],
        3: [w / 2, (w * 3) / 2, h / 2, (h * 3) / 2],
        4: [(w * 3) / 2, w, (h * 3) / 2, h],}
    return  img[int(dict[num][0]) : int(dict[num][1]), int(dict[num][2]) : int(dict[num][3])]

def grep(pattern,word_list):
    expr = re.compile(pattern)
    return [elem for elem in word_list if expr.match(elem)]

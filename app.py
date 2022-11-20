import streamlit as st
import numpy as np
import cv2
import os
import base64
import uuid
import re
import torch
import time
from io import BytesIO
import argparse
from pathlib import Path
from PIL import Image, ImageEnhance
import easyocr


@st.cache
def download_button(object_to_download, download_filename, button_text, isPNG):
    """
    Generates a link to download the given object_to_download.

    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.

    Returns:
    -------
    (str): the anchor tag to download object_to_download

    Examples:
    --------
    download_link(Pillow_image_from_cv_matrix, 'your_image.jpg', 'Click to me to download!')
    """

    buffered = BytesIO()
    if isPNG:
        object_to_download.save(buffered, format="PNG")
    else:
        object_to_download.save(buffered, format="JPEG")
    b64 = base64.b64encode(buffered.getvalue()).decode()
    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 

            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(165,42,42);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'
    return dl_link










def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)



def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='Model_n_notebook/best.pt', force_reload=True)
    return model
model = load_model()

easyOcr = easyocr.Reader(['th'],gpu=True)





def imageInput(license_opt,path_img):
    if license_opt == 'Normal':
        image_list = []
        pred = model(path_img)
        df = pred.pandas().xyxy[0]
        if df.empty :
            return 'There is no license plate detected'
        else:
            for i in range(len(df)):
                row = df.iloc[[i]]
                img_cropped = Image.open(path_img).convert('RGB')
                img_cropped = np.asarray(img_cropped)

                img_cropped = img_cropped[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax'])]

                # image_list.append(np.asarray(im_output))
                im_out = Image.fromarray(img_cropped)
                image_list.append(im_out)

            return image_list

    else:
        image_list = []
        pred = model(path_img)
        df = pred.pandas().xyxy[0]
        if df.empty :
            return 'There is no license plate detected'
        else :
            for i in range(len(df)):
                row = df.iloc[[i]]
                img_cropped = Image.open(path_img).convert('RGB')
                img_cropped = np.asarray(img_cropped)

                img_cropped = img_cropped[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax'])]

                img_equ = cv2.GaussianBlur(img_cropped, (3, 3), 1)

                img_equ =  Image.fromarray(img_equ)
                enhancer = ImageEnhance.Contrast(img_equ)

                factor = 6 #increase contrast
        
                im_output = enhancer.enhance(factor)
                # image_list.append(np.asarray(im_output))
                # im_out = Image.fromarray(im_output)
                image_list.append(im_output)

            return image_list




def main():

    uploaded_file = st.sidebar.file_uploader("Upload your cars image", type=["jpg", "jpeg", 'png'])


    if uploaded_file is not None:
        is_valid = True
        with st.spinner(text='Resource loading...'):
            st.sidebar.image(uploaded_file,caption=f"Original Image")
            with open(os.path.join("images/",uploaded_file.name),"wb") as f:
                f.write(uploaded_file.getbuffer())
                upload_path = f'images/{uploaded_file.name}'
     
    else:
        is_valid = False


    if is_valid:
        if st.sidebar.button('Start'):
            result = model(upload_path)
            result.save()
            for img in os.listdir(get_detection_folder()):
                st.image(str(Path(f'{get_detection_folder()}') / img), use_column_width='always')

            if type(imageInput(license_opt,upload_path)) == str:
                st.write(imageInput(license_opt,upload_path))
            else :
                list_of_image = list(imageInput(license_opt,upload_path))
            
        
                if list_of_image is not None:
                    for i in range(len(list_of_image)):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(list_of_image[i],caption = f'Image No.{i+1}' )
                            st.markdown(download_button(list_of_image[i],f"your_output_file.jpg", f"Save Image No.{i+1}",isPNG = False), unsafe_allow_html=True)
                        with col2:
                            ocr_out = easyOcr.readtext(np.asarray(list_of_image[i]))
                            if len(ocr_out) >=1:
                                st.subheader('License Plate :')
                                st.write(ocr_out[0][1])
                            else:
                                st.subheader('License Plate :')
                                st.write('Can\'t detect any character... \n\n We will improve this in V.1.1')
                st.snow()
                            

            

if __name__ == '__main__':


    st.header("🚗📷Thai - ALPR system V.1.0")
    st.subheader('👈🏽 Select options left-handed menu bar.')
    st.sidebar.title('⚙️Options')
    license_opt = st.sidebar.radio("Select license plate option ", ['Normal', 'Contrast'])
    main()
    


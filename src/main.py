# required packages
import os
import pandas as pd
import pdf2image
import cv2
import pytesseract
import re
from datetime import datetime
import base64
import numpy as np
import logging
from time import process_time
# import utils
import src.utils as utils
import src.rules as rules
from PIL import Image
from fastapi import FastAPI, Body, Form




class BasicExtract:
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', filemode='w')

    def convert_pdf(self, file):
        """
         This function take one argument as
        input. this function will convert
        byte PDF to binary image
        :param file: file
        :return: images
        """
        # Store Pdf with convert_from_path function
        images = pdf2image.convert_from_bytes(file, poppler_path=utils.PROPPLER_PATH, first_page=1, last_page=15, dpi=900)
        logging.info('image convertion done')

        return images

    def remove_dotted_lines(self,images):
        image_list=[]
        for image in images:
            openCVim = np.array(image)
            im_template = cv2.cvtColor(openCVim, cv2.COLOR_RGB2GRAY)

            _, blackAndWhite = cv2.threshold(im_template, 100, 255, cv2.THRESH_BINARY_INV)

            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, None, None, None, 8,
                                                                                 cv2.CV_32S)
            sizes = stats[1:, -1]  # get CC_STAT_AREA component
            img2 = np.zeros((labels.shape), np.uint8)

            for i in range(0, nlabels - 1):
                if sizes[i] >= 100:  # filter small dotted regions
                    img2[labels == i + 1] = 255

            res = cv2.bitwise_not(img2)
            im_pil = Image.fromarray(res)

            image_list.append(im_pil)

        return image_list


    def pre_processing(self, image):
        """
        This function take one argument as
        input. this function will convert
        input image to binary image
        :param image: image
        :return: thresholded image
        """
        dst = cv2.detailEnhance(np.array(image), sigma_s=20, sigma_r=0.1)
        gray_image = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        # converting it to binary image
        threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imwrite('threshold_img.jpg', threshold_img)

        cv2.waitKey(0)

        cv2.destroyAllWindows()
        # logging.info('Thresold Image Convertion done')

        return threshold_img

    def parse_text(self, threshold_img):
        """
        This function take one argument as
        input. this function will feed input
        image to tesseract to predict text.
        :param threshold_img: image
        return: meta-data dictionary
        """
        # configuring parameters for tesseract
        pytesseract.pytesseract.tesseract_cmd = utils.TESSERACT_EXE_PATH
        tesseract_config = r'--tessdata-dir' + utils.TESS_DATA_PATH + '-l eng --oem 3 --psm 11'
        # now feeding image to tesseract
        details = pytesseract.image_to_data(threshold_img, output_type=pytesseract.Output.DICT,
                                            config=tesseract_config, lang='eng')

        # logging.info('Image to text convertion Done')
        return details

    def format_text(self, details):
        """
        This function take one argument as
        input.This function will arrange
        resulted text into proper format.
        :param details: dictionary
        :return: list
        """
        parse_text = []
        word_list = []
        last_word = ''
        for word in details['text']:
            if word != '':
                word_list.append(word)
                last_word = word
            if (last_word != '' and word == '') or (word == details['text'][-1]):
                parse_text.append(word_list)
                word_list = []


        return parse_text

    def basic_data(self, text):
        """
        This function take one argument as
        a input.This function will extract all the
        data from text.
        @param text: All the extracted data from pdf
        @return: All the required data elements
        """
        extra = text

        data = dict()
        for i in rules.list_of_prefix:
            for key, items in rules.list_data.items():
                for j in items:
                    pattern = re.compile(r"" + i + '\s*' + j, re.I)
                    x = set(pattern.findall(extra))
                    for res in x:
                        if res != set():
                            if key in data:
                                data[key].append(res)
                                break
                            else:
                                data[key] = [res]
                                break


        # for key, items in vin_config.list_data2.items():
        #     for i in items:
        #         pattern = re.compile(r"" + i)
        #         x = set(pattern.findall(extra))
        #         for res in x:
        #             if res != set():
        #                 if key in data:
        #                     data[key].append(res)
        #                 else:
        #                     data[key] = [res]

        try:
            if data['Policy Number']:
                pass
        except:
            for key, items in rules.list_data_copy.items():
                for i in items:
                    pattern = re.compile(r""+'\s' + i)
                    x = set(pattern.findall(extra))
                    for res in x:
                        if res != set():
                            if key in data:
                                data[key].append(res)
                                break
                            else:
                                data[key] = [res]
                                break
                    # data[key] = list(set(data[key]))
        if 'Policy Number' in data.keys():
            for indx,p_no in enumerate(data['Policy Number']):
                data['Policy Number'][indx] = data['Policy Number'][indx].strip().replace('#',':').split(':')[-1].strip()

        if 'Policy Number' in data.keys():
            if len(data['Policy Number']) > 1:
                for indx,p_no in enumerate(data['Policy Number']):
                    data['Policy Number'][indx] = data['Policy Number'][indx].strip().replace('#', ':').replace('Policy', ':').replace('policy', ':').replace('Number',':').replace('number', ':').split(':')[-1].strip()

        if 'Policy Number' in data.keys():
            data['Policy Number'] = list(set(data['Policy Number']))

        f_date = list()
        for typee in range(len(rules.date_list['Date'])):
            pattern = re.compile(r'' + rules.date_list['Date'][typee], re.I)
            x = set(pattern.findall(extra))
            x = list(x)
            if x:
                if len(x) == 1:
                    try:
                        data['Effective Date'] = str(pd.to_datetime(x[0]))
                    except:
                        pass
                else:
                    for dtt in x:
                        try:
                            if rules.date_list['Date'][typee] == rules.date_list['Date'][0]:
                                date_time_obj = datetime.strptime(dtt, '%d-%m-%Y')
                            if rules.date_list['Date'][typee] == rules.date_list['Date'][1]:
                                date_time_obj = datetime.strptime(dtt, '%m-%d-%y')
                            if rules.date_list['Date'][typee] == rules.date_list['Date'][2]:
                                date_time_obj = datetime.strptime(dtt, '%b %d, %Y')
                            if rules.date_list['Date'][typee] == rules.date_list['Date'][3]:
                                date_time_obj = datetime.strptime(dtt, '%b %d,%Y')
                            if rules.date_list['Date'][typee] == rules.date_list['Date'][4] or \
                                    rules.date_list['Date'][typee] == rules.date_list['Date'][7]:
                                date_time_obj = datetime.strptime(dtt, '%m/%d/%Y')
                            if rules.date_list['Date'][typee] == (rules.date_list['Date'][5]):
                                date_time_obj = pd.to_datetime(dtt)
                            if rules.date_list['Date'][typee] == rules.date_list['Date'][6]:
                                for dt_format in ('%b %d,%Y','%b %d, %Y'):
                                    try:
                                        date_time_obj = datetime.strptime(dtt, dt_format)
                                    except:
                                        pass

                            f_date.append(date_time_obj)
                        except:
                            pass
        try:
            if f_date[0] < f_date[1]:
                data['Effective Date'] = str(f_date[0])
                data['Expiration Date'] = str(f_date[1])
            if f_date[0] > f_date[1]:
                data['Effective Date'] = str(f_date[1])
                data['Expiration Date'] = str(f_date[0])
        except Exception as e:
            logging.info(e)

        _frequency = dict()
        for key, items in rules.product_type_1.items():
            _pattern_length = 0
            for i in items:
                pattern = re.compile(r'' + i, re.I)
                length = len(pattern.findall(extra))
                _pattern_length += length
            _frequency[key] = _pattern_length

        _prod_type = max(_frequency, key=_frequency.get)

        if _prod_type :
            data['Product Type'] = _prod_type

        _prod_frequency = dict()
        for key, items in rules.Insurer_dict.items():
            _pattern_length = 0
            for i in items:
                pattern = re.compile(r'' + i, re.I)
                length = len(pattern.findall(extra))
                _pattern_length += length
            _prod_frequency[key] = _pattern_length

        _provider = max(_prod_frequency, key=_prod_frequency.get)

        if _prod_type :
            data['Provider Name'] = _provider

        # for key, items in insurer_config.Insurer_dict.items():
        #     for i in items:
        #         pattern = re.compile(r"" + i)
        #         x = list(set(pattern.findall(extra)))
        #         if x:
        #             data['Provider Name'] = key
        #             break

        all_keys = ['Policy Number', 'Effective Date', 'Expiration Date', 'Provider Name']

        for key_val in all_keys:
            if key_val in data.keys():
                pass
            else:

                data[key_val] = 'Not found'
        if 'Product Type' not in data.keys():
            data['Product Type'] = 'Code Issue'

        logging.info('Data extraction Done')
        return data
    def get_text(self, images):
        temp_text = list()
        for i in images:
            # calling pre_processing function to perform pre-processing on input image.
            # thresholds_image = self.pre_processing(i)
            # calling parse_text function to get text from image by Tesseract.
            #calling remove dots function to remove dotted lines
            # clean_image=self.remove_dots(i)
            parsed_data = self.parse_text(i)
            # calling format_text function which will format text according to input image
            arranged_text = self.format_text(parsed_data)
            temp_text += arranged_text
        # txt = ' '.join(temp_text)
        txt = ''
        for list_data in temp_text:
            for word in list_data:
                txt += word + ' '
        # write_text(temp_text)
        logging.info('Processing Done')
        return txt

    def main(self, byte):
        try:
            imgname = self.convert_pdf(byte)
        except Exception as e:
            logging.error(e)
        else:
            try:
                text = self.get_text(imgname)
            except Exception as e:
                logging.info('{} text extraction faild with error'.format(e))
            else:
                try:
                    exd = self.basic_data(text)
                    logging.info(exd)
                except Exception as e:
                    logging.error(e)
        return exd

def main(encoded_string):
    """
    Main function executes script
    Returns:
      None
    """

    class_init = BasicExtract()
    extracted_list = list()
    byte = base64.b64decode(encoded_string, validate=True)
    # start time
    t1_start = process_time()

    data = class_init.main(byte)

    t1_stop = process_time()
    elapsed_time = t1_stop - t1_start
    data['Elapsed_time'] = elapsed_time
    return data

app = FastAPI()

@app.post("/get_data")
async def read_item(string : str = Body(...)):
    string = string
    return {"item_id": main(string)}




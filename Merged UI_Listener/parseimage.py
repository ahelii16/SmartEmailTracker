"""
Created on Wed Jun 24

@author: divya
"""
# install tesseract first
# from here
# https://github.com/tesseract-ocr/tesseract/wiki#installation
# then install pillow
# pip install Pillow
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract


def ocr_core(filename):

    text = pytesseract.image_to_string(Image.open(filename))
    # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    #print(f'Extracted from image {text}')
    return text

#print(ocr_core('images/ocr_example_1.png'))

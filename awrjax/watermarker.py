import numpy as np
from PIL import Image, ImageDraw, ImageFont


def addtive_wartermark(img, watermark, alpha):
    return img * (1 - alpha) + watermark * alpha


def generate_text_image(text,
                        font_path,
                        bg_color=(0, 0, 0),
                        fg_color=(255, 255, 255),
                        font_size=200):
    # Define the image font and size
    # You can use fc-list :lang=zh to find the path of the Chinese font you want to use
    font = ImageFont.truetype(font=font_path, size=font_size)

    # Create an image with white background
    image = Image.new("RGB", (font_size * len(text), font_size), bg_color)
    draw = ImageDraw.Draw(image)

    # Get the size text will take up and create a new image of that size
    text_size = draw.textsize(text, font=font)
    text_size = (text_size[0] + 10, text_size[1] + 20)
    image = Image.new("RGB", text_size, bg_color)
    draw = ImageDraw.Draw(image)

    # Draw the text on the image
    draw.text((0, 0), text, fg_color, font=font, spacing=0)

    return np.asarray(image)

from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_image_array_from_text(text, size=(100, 100)):
    # image size and bgcolor
    width, height = size
    background_color = 0

    # make new img file
    image = Image.new("L", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    # font
    font_size = size[0]
    font = ImageFont.truetype("arial.ttf", font_size) 

    text_bbox = draw.textbbox((0, 0), text, font=font)

    # drawing in the middle
    x = (width - (text_bbox[2] - text_bbox[0])) // 2 - text_bbox[0]
    y = (height - (text_bbox[3] - text_bbox[1])) // 2 - text_bbox[1]
    draw.text((x, y), text, fill=255, font=font) 
    
    image_array = np.array(image)

    return image_array

# 사용자 입력 받기
user_input = input("이미지로 변환할 문자를 입력하세요: ")

# 흑백 이미지 배열 생성
image_array = create_image_array_from_text(user_input, size=(12, 12))
print("흑백 이미지 배열:")
print(image_array)
print("이미지 배열의 형태:", image_array.shape)

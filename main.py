import cv2
import numpy as np
import re
from pytesseract import Output, image_to_data

custom_config = r'--oem 3 --psm 6'
data_pattern_letters = '^([A-Z][A-Z][A-Z])$|\W[A-Z][A-Z][A-Z]|[A-Z][A-Z][A-Z]\W'
data_pattern_numbers = '^([0-9][0-9][0-9])$|\W[0-9][0-9][0-9]|[0-9][0-9][0-9]\W'


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def remove_noise(image):
    return cv2.medianBlur(image, 5)


def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def image_preprocessing(image, tries):
    if tries == 1:
        image = get_grayscale(image)
    elif tries == 2:
        image = thresholding(image)
    elif tries == 3:
        image = erode(image)
    elif tries == 4:
        image = opening(image)
    elif tries == 5:
        image = deskew(image)
    tries += 1
    return image, tries


def get_tesseract_data(image):
    ocr_data = image_to_data(image, config=custom_config, output_type=Output.DICT)
    print(ocr_data)
    return ocr_data


def search_tesseract_data(data, text_pattern, num_pattern, letters, numbers):

    index = 0
    lx, ly, nx, ny, nw, nh = 0, 0, 0, 0, 0, 0
    for i in data['text']:
        if re.search(text_pattern, i) and data['conf'][index] > 50:
            letters = i
            lx, ly, lw, lh = data['left'][index], data['top'][index], data['width'][
                index], data['height'][index]
        elif re.search(num_pattern, i) and data['conf'][index] > 50:
            numbers = i
            nx, ny, nw, nh = data['left'][index], data['top'][index], data['width'][
                index], data['height'][index]
        index += 1

    if lx == 0:
        lx = nx
        ly = ny
    elif ly == 0:
        lx = nx
        ly = ny

    rectangle_pos1 = (lx, ly)
    rectangle_pos2 = (nx + nw, ny + nh)
    # letters = re.sub(r'[^A-Z]', '', letters)
    letters = re.sub(r'\W', '', letters)
    # numbers = re.sub(r'[^0-9]', '', numbers)
    numbers = re.sub(r'\W', '', numbers)

    return letters, numbers, rectangle_pos1, rectangle_pos2


def draw_license_plate_pos(image, letters, numbers, rectangle_pos1, rectangle_pos2):
    image = cv2.rectangle(image, rectangle_pos1, rectangle_pos2, (0, 255, 0), 14)
    image = cv2.putText(image, f'License plate {letters} {numbers} found', (rectangle_pos1[0] - 450,
                                                                            rectangle_pos1[1] + 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5, cv2.LINE_AA)
    image = cv2.putText(image, f'License plate {letters} {numbers} found', (rectangle_pos1[0] - 450,
                                                                            rectangle_pos1[1] + 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

    cv2.imshow('img', image)
    cv2.waitKey(0)

    return


def main():
    img = cv2.imread('bil2.jpg')
    img2 = cv2.imread('bil2.jpg')
    attempt = 1
    done = False
    letters = 'NOTHING'
    numbers = ''
    while not done:
        img2, attempt = image_preprocessing(img2, attempt)
        tesseract_data = get_tesseract_data(img2)
        license_plate_letters, license_plate_numbers, coord1, coord2 = search_tesseract_data(
            tesseract_data, data_pattern_letters, data_pattern_numbers, letters, numbers)
        if (re.match('^[A-Z][A-Z][A-Z]$', license_plate_letters) and re.match('^[0-9][0-9][0-9]$', license_plate_numbers)) or attempt > 6:
            done = True
    draw_license_plate_pos(img, license_plate_letters, license_plate_numbers, coord1, coord2)


if __name__ == '__main__':
    main()

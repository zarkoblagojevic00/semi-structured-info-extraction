import pyocr.builders
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from face import FaceExtractor
from datetime import datetime


class Person:
    """
    Klasa koja opisuje prepoznatu osobu sa slike. Neophodno je prepoznati samo vrednosti koje su opisane u ovoj klasi
    """
    def __init__(self, name: str = None, date_of_birth: datetime.date = None, job: str = None, ssn: str = None,
                 company: str = None):
        self.name = name
        self.date_of_birth = date_of_birth
        self.job = job
        self.ssn = ssn
        self.company = company


# https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
def rotate(img, angle, center=None):
    width, height = img.shape[1], img.shape[0]
    if center is None:
        center = (width // 2, height // 2)
    rot_angle = 90 + angle if angle < -45 else angle
    rotation_M = cv2.getRotationMatrix2D(center, rot_angle, 1)
    return cv2.warpAffine(img, rotation_M, (width, height), flags=cv2.INTER_CUBIC)


def get_min_area_rect_data(contour):
    center, size, angle = cv2.minAreaRect(contour)
    width, height = size
    if height > width:
        higher = height
        lower = width
    else:
        higher = width
        lower = height
    return center, angle, higher, lower


# https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
def fix_contrast(img_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)

def remove_non_alphan(word: str):
    return ''.join(c for c in word if c.isalnum() or c in ["-", "," "."])

def capitalize_words(words: str):
    return ' '.join([remove_non_alphan(word).capitalize() if not word.isupper() else word for word in words.split(' ')])

def parse_date(date: str):
    format = '%d %b %Y'
    try:
        return datetime.strptime(date, format)
    except Exception:
        print(f'Invalid date: {date}')
        return datetime(2020, 12, 23)


# https: // docs.opencv.org / 3.4 / dc / da3 / tutorial_copyMakeBorder.html
def make_black_border_img(img):
    width, height = img.shape[1], img.shape[0]
    percent = 0.05
    top = int(3 * percent * height)
    bottom = top
    left = int(percent * width)
    right = left
    value = cv2.mean(img)

    border = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=value)

    return border



class Document:
    def __init__(self, image_path: str, ocr_tool,  extractor):
        print(image_path)
        self.ocr_tool = ocr_tool
        self.image = cv2.imread(image_path)
        self.lang = 'eng'
        self.extractor = extractor

    def read_person_data(self):
        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        card = self.crop_apple_card(img_hsv)
        if card is not None:
            card, orange_part_height, orange_part_width = card
            return self.extract_apple_person(card, orange_part_height, orange_part_width)

        card = self.crop_google_card(img_hsv)
        if card is not None:
            return self.extract_google_person(card)

        card = self.crop_ibm_card(img_hsv)
        return self.extract_ibm_person(card)





    def crop_apple_card(self, img_hsv):
        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        mask_apple = cv2.inRange(img_hsv, np.array([20, 190, 0]), np.array([21, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        mask_apple_eroded = cv2.morphologyEx(mask_apple, cv2.MORPH_ERODE, kernel)
        _, contours, _ = cv2.findContours(mask_apple_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
            return None

        potential_orange_sides = []
        for contour in contours:

            _, _, higher, lower = get_min_area_rect_data(contour)
            if higher > 15 and lower > 15:
                aspect_ratio = lower / float(higher)
                if 0.10 < aspect_ratio < 0.35:
                    potential_orange_sides.append(contour)

        if len(potential_orange_sides) < 1:
            return None
        orange_side = max(potential_orange_sides, key=lambda cnt: cv2.contourArea(cnt))

        # apple_cnt = cv2.drawContours(self.image.copy(), [orange_side], 0, (0, 255, 0), 3)
        # plt.imshow(apple_cnt)
        # plt.show()

        center, angle, higher, lower = get_min_area_rect_data(orange_side)
        img_rotated = rotate(self.image, angle, center)

        x, y = center
        img_cropped = img_rotated[int(y - higher / 2):int(y + higher / 2), int(x - lower / 2): int(x + 8 * lower)]
        # plt.imshow(img_cropped)
        # plt.show()

        img_cropped_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        return \
            img_cropped if self.extractor.is_face_on_img(img_cropped_gray) else None, \
            higher, \
            lower

    def crop_google_card(self, img_hsv):
        mask_google = cv2.inRange(img_hsv, np.array([85, 125, 0]), np.array([91, 255, 255]), )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_google_eroded = cv2.morphologyEx(mask_google, cv2.MORPH_ERODE, kernel)

        _, contours, _ = cv2.findContours(mask_google_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
            return None

        potential_google_cards = []
        for contour in contours:

            _, _, higher, lower = get_min_area_rect_data(contour)
            if higher > 15 and lower > 15:
                aspect_ratio = lower / float(higher)
                if 0.25 < aspect_ratio < 0.5:
                    potential_google_cards.append(contour)

        if len(potential_google_cards) < 1:
            return None
        google_card = max(potential_google_cards, key=lambda cnt: cv2.contourArea(cnt))

        # google_cnt = cv2.drawContours(self.image.copy(), [google_card], 0, (0, 255, 0), 3)
        # plt.imshow(google_cnt)
        # plt.show()

        center, angle, higher, lower = get_min_area_rect_data(google_card)
        img_rotated = rotate(self.image, angle, center)

        x, y = center
        img_cropped = img_rotated[int(y - 2.6 * lower / 2):int(y + 2.3 * lower / 2 - 10),
                      int(x - higher / 2): int(x + 2.1 * higher / 2)]
        # plt.imshow(img_cropped)
        # plt.show()
        img_cropped_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        return img_cropped if self.extractor.is_face_on_img(img_cropped_gray) else None

    def crop_ibm_card(self, img_hsv):
        mask_ibm = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([1, 255, 255]), )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))
        mask_ibm = cv2.morphologyEx(mask_ibm, cv2.MORPH_ERODE, kernel)

        img_width, img_height = img_hsv.shape[1], img_hsv.shape[0]
        mask_ibm[0: int(img_height * 0.09), :] = 0
        mask_ibm[img_height - int(img_height * 0.09):, :] = 0

        _, contours, _ = cv2.findContours(mask_ibm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
            return None

        potential_ibm_cards = []
        for contour in contours:

            _, _, higher, lower = get_min_area_rect_data(contour)
            if higher > 15 and lower > 15:
                aspect_ratio = lower / float(higher)
                if 0.5 < aspect_ratio < 0.75:
                    potential_ibm_cards.append(contour)

        if len(potential_ibm_cards) < 1:
            return None
        ibm_card = max(potential_ibm_cards, key=lambda cnt: cv2.contourArea(cnt))

        # ibm_cnt = cv2.drawContours(self.image.copy(), [ibm_card], 0, (0, 255, 0), 3)
        # plt.imshow(ibm_cnt)
        # plt.show()

        center, angle, higher, lower = get_min_area_rect_data(ibm_card)
        img_rotated = rotate(self.image, angle, center)

        x, y = center
        img_cropped = img_rotated[int(y - lower / 2 + 10):int(y + lower / 2 - 10),
                      int(x - higher / 2 + 10): int(x + higher / 2 - 10)]
        # plt.imshow(img_cropped)
        # plt.show()

        img_cropped_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        return img_cropped if self.extractor.is_face_on_img(img_cropped_gray) else None


    def extract_apple_person(self, card, orange_part_height, orange_part_width):

        ################################################# SSN #########################################################
        card_hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
        mask_apple = cv2.inRange(card_hsv, np.array([20, 190, 0]), np.array([21, 255, 255]))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 15))
        mask_apple = cv2.morphologyEx(mask_apple, cv2.MORPH_CLOSE, kernel)

        # plt.imshow(mask_apple, 'gray')
        # plt.show()

        _, contours, _ = cv2.findContours(mask_apple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 1:
            return None

        ssn_boxes = []
        for contour in contours:

            _, _, higher, lower = get_min_area_rect_data(contour)
            if higher > 30:
                ssn_boxes.append(contour)

        if len(ssn_boxes) < 1:
            return None
        ssn_box = max(ssn_boxes, key=lambda cnt: cv2.boundingRect(cnt)[2])

        x, y, w, h = cv2.boundingRect(ssn_box)
        ssn_box_crop = card[y-int(h*0.3): y+int(h*1.3), x-10: x+w+10]
        # plt.imshow(ssn_box_crop)
        # plt.show()

        # https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
        ssn_box_crop = make_black_border_img(fix_contrast(cv2.cvtColor(ssn_box_crop, cv2.COLOR_BGR2GRAY)))
        _, ssn_bin = cv2.threshold(ssn_box_crop, 0, 255, cv2.THRESH_OTSU)

        # plt.imshow(ssn_bin, 'gray')
        # plt.show()

        ssn_boxes = self.ocr_tool.image_to_string(
            Image.fromarray(ssn_bin), lang=self.lang,
            builder=pyocr.builders.LineBoxBuilder(tesseract_layout=7)
        )

        ssn = ""
        for i, line in enumerate(ssn_boxes):
            ssn = capitalize_words(line.content.strip())
            print('line %d: ' % i, ssn, line.position)
        print()

        # plt.imshow(card)
        # plt.show()

        elem_height = int(w*0.165)

        ################################################# SSN #########################################################
        pointer = y - 3*elem_height
        date_of_birth_box_crop = card[pointer-int(h*0.4): pointer+int(h*1.6), x-int(w*0.5): x+int(w*1.2)]


        date_of_birth_box_crop = make_black_border_img(fix_contrast(cv2.cvtColor(date_of_birth_box_crop, cv2.COLOR_BGR2GRAY)))
        _, date_of_birth_bin = cv2.threshold(date_of_birth_box_crop, 0, 255, cv2.THRESH_OTSU)

        # plt.imshow(date_of_birth_bin, 'gray')
        # plt.show()

        date_of_birth_boxes = self.ocr_tool.image_to_string(
            Image.fromarray(date_of_birth_bin), lang=self.lang,
            builder=pyocr.builders.LineBoxBuilder(tesseract_layout=7)
        )

        date_of_birth = ""
        for i, line in enumerate(date_of_birth_boxes):
            date_of_birth = capitalize_words(line.content.strip())
            print('line %d: ' % i, date_of_birth, line.position)
        print()

        ################################################# NAME #########################################################
        pointer -= 3 * elem_height
        name_box_crop = card[pointer - int(h * 0.4): pointer + int(h * 1.6),
                                 x - int(w * 0.5): x + int(w * 1.2)]

        # plt.imshow(name_box_crop)
        # plt.show()

        name_box_crop = make_black_border_img(
            fix_contrast(cv2.cvtColor(name_box_crop, cv2.COLOR_BGR2GRAY)))
        _, name_bin = cv2.threshold(name_box_crop, 0, 255, cv2.THRESH_OTSU)

        # plt.imshow(name_bin, 'gray')
        # plt.show()

        name_boxes = self.ocr_tool.image_to_string(
            Image.fromarray(name_bin), lang=self.lang,
            builder=pyocr.builders.LineBoxBuilder(tesseract_layout=7)
        )

        name = ""
        for i, line in enumerate(name_boxes):
            name = capitalize_words(line.content.strip())
            print('line %d: ' % i, name, line.position)
        print()

        ################################################# JOB #########################################################
        pointer -= elem_height
        job_box_crop = card[pointer - int(h * 0.4): pointer + int(h * 1.6),
                        x - int(w * 0.5): x + int(w * 1.2)]

        # plt.imshow(job_box_crop)
        # plt.show()

        job_box_crop = make_black_border_img(
            fix_contrast(cv2.cvtColor(job_box_crop, cv2.COLOR_BGR2GRAY)))
        _, job_bin = cv2.threshold(job_box_crop, 0, 255, cv2.THRESH_OTSU)

        # plt.imshow(job_bin, 'gray')
        # plt.show()

        job_boxes = self.ocr_tool.image_to_string(
            Image.fromarray(job_bin), lang=self.lang,
            builder=pyocr.builders.LineBoxBuilder(tesseract_layout=7)
        )

        job = ""
        for i, line in enumerate(job_boxes):
            job = capitalize_words(line.content.strip())
            print('line %d: ' % i, job, line.position)
        print()

        company = 'Apple'

        return Person(name, parse_date(date_of_birth), job, ssn, company)

    def extract_google_person(self, card):
        card_hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
        mask_google = cv2.inRange(card_hsv, np.array([85, 75, 0]), np.array([92, 255, 255]), )
        contrasted = fix_contrast(cv2.cvtColor(card, cv2.COLOR_BGR2GRAY))
        inv = 255 - contrasted

        width, height = card.shape[1], card.shape[0]
        crop_info = inv[int(height/5): height - int(height/4.25), int(width/28): width - int(width/2.6)]
        if crop_info.shape[0] < 200:
            crop_info = cv2.resize(crop_info, (350, 200), cv2.INTER_CUBIC)

        # plt.imshow(crop_info, 'gray')
        # plt.show()

        info_w, info_h = crop_info.shape[1], crop_info.shape[0]
        crop_name = crop_info[0:int(info_h/5), :]
        # plt.imshow(crop_name, 'gray')
        # plt.show()
        # self.hsv_picker(card)

        name_boxes = self.ocr_tool.image_to_string(
            Image.fromarray(crop_name), lang=self.lang,
            builder=pyocr.builders.LineBoxBuilder(tesseract_layout=7)
        )

        name = ""
        for i, line in enumerate(name_boxes):
            name = capitalize_words(line.content.strip())
            print('line %d: ' % i, name, line.position)
        print()

        crop_rest = crop_info[int(info_h / 5):, int(info_w/2.4):]

        rest_height, rest_width = crop_rest.shape[0], crop_rest.shape[1]
        ideal_height = 160
        aspect = ideal_height/rest_height
        if rest_height > ideal_height:
            crop_rest = cv2.resize(crop_rest, (int(aspect * rest_width), ideal_height), cv2.INTER_CUBIC)

        rest_height, rest_width = crop_rest.shape[0], crop_rest.shape[1]
        # plt.imshow(crop_rest, 'gray')
        # plt.show()

        ################################################# SSN #########################################################
        dist = int(rest_height / 5.2)

        base_height = int(rest_height / 6.8)
        ssn_height = base_height + dist

        crop_ssn = crop_rest[base_height:ssn_height, :]
        crop_ssn = make_black_border_img(crop_ssn)
        # plt.imshow(crop_ssn, 'gray')
        # plt.show()

        ssn_boxes = self.ocr_tool.image_to_string(
            Image.fromarray(crop_ssn), lang=self.lang,
            builder=pyocr.builders.LineBoxBuilder(tesseract_layout=7)
        )

        ssn = ""
        for i, line in enumerate(ssn_boxes):
            ssn = capitalize_words(line.content.strip())
            print('line %d: ' % i, ssn, line.position)
        print()

        ################################################# JOB #########################################################
        job_height = ssn_height + dist
        crop_job = crop_rest[ssn_height: job_height, :]
        crop_job = make_black_border_img(crop_job)

        # plt.imshow(crop_job, 'gray')
        # plt.show()

        job_boxes = self.ocr_tool.image_to_string(
            Image.fromarray(crop_job), lang=self.lang,
            builder=pyocr.builders.LineBoxBuilder(tesseract_layout=7)
        )

        job = ""
        for i, line in enumerate(job_boxes):
            job = capitalize_words(line.content.strip())
            print('line %d: ' % i, job, line.position)
        print()

        ################################################# DOB #########################################################
        date_of_birth_height = job_height + dist
        crop_date_of_birth = crop_rest[job_height: date_of_birth_height, :]
        crop_date_of_birth = make_black_border_img(crop_date_of_birth)
        # plt.imshow(crop_date_of_birth, 'gray')
        # plt.show()

        date_of_birth_boxes = self.ocr_tool.image_to_string(
            Image.fromarray(crop_date_of_birth), lang=self.lang,
            builder=pyocr.builders.LineBoxBuilder(tesseract_layout=7)
        )

        date_of_birth = ""
        for i, line in enumerate(date_of_birth_boxes):
            date_of_birth = capitalize_words(line.content.strip())
            print('line %d: ' % i, date_of_birth, line.position)
        print()

        company = 'Google'
        return Person(name, parse_date(date_of_birth), job, ssn, company)










    def extract_ibm_person(self, card):
        pass




    def hsv_picker(self, image):
        def nothing(x):
            pass

        # Load image

        # Create a window
        cv2.namedWindow('image')

        # Create trackbars for color change
        # Hue is from 0-179 for Opencv
        cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
        cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
        cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
        cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
        cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
        cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

        # Set default value for Max HSV trackbars
        cv2.setTrackbarPos('HMax', 'image', 179)
        cv2.setTrackbarPos('SMax', 'image', 255)
        cv2.setTrackbarPos('VMax', 'image', 255)

        # Initialize HSV min/max values
        hMin = sMin = vMin = hMax = sMax = vMax = 0
        phMin = psMin = pvMin = phMax = psMax = pvMax = 0

        while (1):
            # Get current positions of all trackbars
            hMin = cv2.getTrackbarPos('HMin', 'image')
            sMin = cv2.getTrackbarPos('SMin', 'image')
            vMin = cv2.getTrackbarPos('VMin', 'image')
            hMax = cv2.getTrackbarPos('HMax', 'image')
            sMax = cv2.getTrackbarPos('SMax', 'image')
            vMax = cv2.getTrackbarPos('VMax', 'image')

            # Set minimum and maximum HSV values to display
            lower = np.array([hMin, sMin, vMin])
            upper = np.array([hMax, sMax, vMax])

            # Convert to HSV format and color threshold
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(image, image, mask=mask)

            # Print if there is a change in HSV value
            if ((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (
                    pvMax != vMax)):
                print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (
                    hMin, sMin, vMin, hMax, sMax, vMax))
                phMin = hMin
                psMin = sMin
                pvMin = vMin
                phMax = hMax
                psMax = sMax
                pvMax = vMax

            # Display result image
            cv2.imshow('image', result)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


apple = [
    0, 3, 7, 14, 15, 21, 23, 24, 26, 30,
    34, 36, 37, 41, 42, 48, 49, 60, 64, 65,
    72, 76, 77, 81, 82, 84, 87, 90, 92, 94,
    96, 101, 109, 112, 116, 117, 118, 124, 126, 127,
    130, 133, 140, 143
]

ibm = [
    2, 4, 8, 13, 17, 20, 22, 27, 28, 31,
    32, 40, 44, 45, 47, 51, 52, 53, 56, 57,
    58, 63, 66, 67, 69, 71, 75, 83, 85, 86,
    88, 89, 91, 97, 98, 99, 102, 103, 105, 106,
    107, 113, 114, 115, 119, 121, 122, 123, 129, 132,
    135, 137, 139, 141, 144, 145, 147
]

google = [i for i in range(0, 150) if i not in ibm and i not in apple]

list = ibm
paths = [f'./dataset/validation/image_{item}.bmp' for item in list]
for path in paths:
    ocr_tool = pyocr.get_available_tools()[0]
    extractor = FaceExtractor()

    # words = ["joshua Chase", "Joshua Chase", "Robert Downey jr.", "Paul McCartney", "debora's Highnes DDS II", "second 2nd son"]
    # for word in words:
    #     print(clean_words(word))
    # dates = ["23 Dec 2020", "Dec 2020", "Dec 202a"]
    # for date in dates:
    #     print(parse_date(date))
    doc = Document(path, ocr_tool, extractor)
    doc.read_person_data()

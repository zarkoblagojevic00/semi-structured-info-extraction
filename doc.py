import pyocr.builders
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from face import FaceExtractor


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
            return self.extract_apple_person(card)

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
        plt.imshow(img_cropped)
        plt.show()

        img_cropped_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        return img_cropped if self.extractor.is_face_on_img(img_cropped_gray) else None

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
        plt.imshow(img_cropped)
        plt.show()
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
        plt.imshow(img_cropped)
        plt.show()

        img_cropped_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        return img_cropped if self.extractor.is_face_on_img(img_cropped_gray) else None






    def extract_apple_person(self, card):
        pass

    def extract_google_person(self, card):
        pass

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

list = google
paths = [f'./dataset/validation/image_{item}.bmp' for item in list]
for path in paths:
    ocr_tool = pyocr.get_available_tools()[0]
    extractor = FaceExtractor()

    doc = Document(path, ocr_tool, extractor)
    doc.read_person_data()

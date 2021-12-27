from imutils import face_utils
import dlib

class FaceExtractor:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        # self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def is_face_on_img(self, img):
        try:
            self.detector(img, 1)[0]
            print("Detected face")
        except Exception:
            print("Face not detected")
            return False
        return True

    def find_face(self, img):
        rects = self.detector(img, 1)
        x, y, w, h = face_utils.rect_to_bb(rects[0])
        face_img = img[y:y+h+1, x:x+w+1]

        return face_img
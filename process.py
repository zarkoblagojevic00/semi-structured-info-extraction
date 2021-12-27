from doc import Document, Person
from face import FaceExtractor
import pyocr
import datetime




def extract_info(image_path: str) -> Person:
    """
    Procedura prima putanju do slike sa koje treba ocitati vrednosti, a vraca objekat tipa Person, koji predstavlja osobu sa licnog dokumenta.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param image_path: <str> Putanja do slike za obradu
    :return: Objekat tipa "Person", gde su svi atributi setovani na izvucene vrednosti
    """
    ocr_tool = pyocr.get_available_tools()[0]
    extractor = FaceExtractor()

    doc = Document(image_path, ocr_tool, extractor)
    person = doc.read_person_data()

    if person is None:
        person = Person('test', datetime.date.today(), 'test', 'test', 'test')

    # TODO - Prepoznati sve neophodne vrednosti o osobi sa slike. Vrednosti su: Name, Date of Birth, Job,
    #       Social Security Number, Company Name

    return person

# ************* IMPORT **************
from lingua import Language, LanguageDetectorBuilder

class LinguaModel:
    """
    Define Lingua model
    """
    def __init__(self):
        self.lingua = self.create_lingua()

    def create_lingua(self):
        language_data = [Language.ENGLISH, Language.FRENCH]
        detector = LanguageDetectorBuilder.from_languages(*language_data).build()
        return detector
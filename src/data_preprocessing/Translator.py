from deep_translator import GoogleTranslator

class Translator():
    def __init__(self, src_lang='en', dest_lang='en'):
        """Initialization of parameters
        
        :param src_lang: source language of tweets
        :param dest_lang: destination language of tweets
        """
        
        self.src_lang = src_lang
        self.dest_lang = dest_lang
        self.translator = GoogleTranslator(source=self.src_lang, target=self.dest_lang)

    def translate(self, text):
        """Translate text
        
        :param text: text for translation
        
        :return: translated text
        """

        try:      
            t = self.translator.translate(text)
            return t

        except Exception as ex:
            print(ex)
            raise ex
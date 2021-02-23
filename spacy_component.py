import logging
import spacy

from spacy_langdetect import LanguageDetector

class SpacyComponent:
    """ Spacy object to process text for various NLP components"""

    current_text = ""  # used to see if its a new text that needs to be doc'd

    def __init__(self, model_path):
        self.nlp = self.load(model_path)
        self.doc = None

    def load(self, model_path):
        """ load a given spacy model


            Keyword arguments:
            model_path -- path to where model is located

            Returns:
            nlp -- spacy pipeline containing language model

        """
        nlp = spacy.load(model_path)
        logging.info("Loaded spacy model at %s", model_path)

        if nlp.has_pipe("parser"):  # language detection works with dependency parsing
            nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
        return nlp

    def doc_spacy(self, input_text):
        """ process text into spacy pipeline and put processed text it in doc"""

        logging.info("Processing text in spacy pipeline...")
        if input_text != self.current_text:
            self.doc = self.nlp(input_text)
            self.current_text = input_text
            logging.debug("Found new text %s", input_text)
        else:
            logging.debug("No new text found, text is still: %s", input_text)

    def namedentityrecognizer(self, input_text):
        """ extract entities from given text


            Keyword arguments:
            text -- sentence that should be processed for NER as a string

            Returns:
            result -- Ner with relative position in sentence, token and its label as a dict

        """
        self.doc_spacy(input_text)
        logging.info("Looking for entities...")
        result = [{"Start": ent.start_char,
                   "End": ent.end_char,
                   "Token": ent.text,
                   "Label": ent.label_}
                  for ent in self.doc.ents]
        logging.debug("Found entities: %s", result)

        return result

    def intentdetector(self, input_text):
        """ extract intents from given text


        Keyword arguments:
        input_text -- sentence that should be processed for intent as a string

        Returns:
        result -- intent with label and certainty

        """
        self.doc_spacy(input_text)

        logging.info("Looking for intent...")
        if len(self.doc.cats) == 0:  # if no intents return empty dict
            return {}
        intent = max(self.doc.cats, key=self.doc.cats.get)
        # if self.doc.cats[intent] < 0.5:
        #     return {[]}
        # else:
        result = {"Label": intent,
                  "Certainty": self.doc.cats[intent]}
        logging.debug("Found intents: %s", self.doc.cats)

        return [result]

    def tokenizer(self, input_text):
        """ tokenize a text


        Keyword arguments:
        input_text -- sentence that should be tokenized

        Returns:
        res -- array of tokens
        """

        self.doc_spacy(input_text)

        result = [{"Token": str(token),
                   "Start": int(token.idx),
                   "End": int(token.idx + len(token))} for token in self.doc]
        logging.debug("Found Token: %s", result)

        return result

    def languagerecognizer(self, input_text):
        """ detects language of text


        Keyword arguments:
        input_text -- sentence that language is checked for

        Returns:
        language -- predicted language
        """
        self.doc_spacy(input_text)

        logging.info("Detecting a language...")
        language = self.doc._.language
        logging.info("Found language: %s", language)

        return language

    def lemmatizer(self, input_text):
        """ gets the lemmatize for every word in a text


        Keyword arguments:
        input_text -- sentence that should be lemmatize

        Returns:
        result -- dict of tokens with their lemmas

        """

        self.doc_spacy(input_text)

        logging.info("Looking for lemma...")
        result = [{"Token": str(token),
                   "Start": int(token.idx),
                   "End": int(token.idx + len(token)),
                   "Lemma": str(token.lemma_)} for token in self.doc]
        logging.debug("Found lemmas: %s", result)
        logging.info("Finished lemmatizing")

        return result

    def postagger(self, input_text):
        """ gets the Part of Speech for every word in a text


        Keyword arguments:
        input_text -- sentence that should be lemmatize

        Returns:
        result -- dict of tokens with their PoS-Tags

        """

        self.doc_spacy(input_text)

        logging.info("Looking for Part of Speech...")
        result = [{"Token": str(token),
                   "Start": int(token.idx),
                   "End": int(token.idx + len(token)),
                   "Dep": str(token.pos_)} for token in self.doc]
        logging.debug("Found POS: %s", result)
        logging.info("Finished Part of Speech tagging")

        return result


if __name__ == "__main__":
    """Using for local testing purposes"""

    testObj = SpacyComponent("olibot")
    text = "Google want me to do this Marcus!"
    print(testObj.namedentityrecognizer(text))
    print(testObj.tokenizer(text))
    print(testObj.postagger(text))
    print(testObj.lemmatizer(text))
    print(testObj.languagerecognizer(text))

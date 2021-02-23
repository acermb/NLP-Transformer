import logging, json, os

from fastapi import FastAPI, HTTPException
import importlib
from monitor import Monitor
from spacy_component import SpacyComponent

from pydantic import BaseModel
SEP = os.path.sep

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    filename="logs.txt",
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    filemode='w', level=logging.DEBUG)
    monitor = Monitor(10)

    nlp = SpacyComponent("de_trf_bertbasecased_lg")

    while True:
        message = input("Please enter something: ")
        print(nlp.namedentityrecognizer(message))

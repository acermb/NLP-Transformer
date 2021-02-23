import logging, random, time, warnings
import json
import spacy
import itertools
import torch

from collections import defaultdict
from pathlib import Path
from spacy.util import minibatch, compounding, decaying
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from spacy_transformers.util import cyclic_triangular_rate


class BertModel:
    """Use transfer learning on a bert based model in spacy
    """
    def __init__(self):
        # initialise model
        logging.info("Initialising spacy model...")
        self.test_data = None # dataset that is used for testing
        self.path = None
        self.id = None
        self.can_retrain = True
        self.hyperparameters = {"namedentityrecognizer_iterations": 5, "namedentityrecognizer_dropout": 0.25,
                                "intentdetector_iterations": 5, "intentdetector_dropout": 0.25} # sets default values for hyperparameters
        self.scores = {}
        self.components = []
        self.entity_score = [[0,0,0]]
        self.cat_scores = [[0,0,0]]
        self.losses_ner = []
        self.losses_cat = []

    def load_model(self, model, train_data):
        """ loads a spacy-bert language model and prepares pipeline
        
        Keywords argument:
        model -- spacy model to load, set to None for a new blank model

        Return:
        nlp -- loaded model

        """

        nlp = spacy.load("de_trf_bertbasecased_lg")
        logging.info("Loaded model %s", str("de_trf_bertbasecased_lg"))

        # create the built-in pipeline components and add them to the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner, last=True)
        # otherwise, get it so we can add labels
        else:
            ner = nlp.get_pipe("ner")

        if "textcat" not in nlp.pipe_names:
            textcat = nlp.create_pipe("trf_textcat", config={"exclusive_classes": True, "architecture": "softmax_last_hidden"})
            nlp.add_pipe(textcat, last=True)
        # otherwise, get it so we can add labels
        else:
            textcat = nlp.get_pipe("trf_textcat")

            # add labels
        for _, annotations, annotations2 in train_data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])
            for cat in annotations2.get("cats"):
                textcat.add_label(cat)

        return nlp

    def train_entity(self, nlp, output_dir, train_data,n_iter, dropout):
        """Load the model, set up the pipeline and train the entity recognizer.
        

        Keyword arguments:
        model -- path to the model if existent
        output_dir -- path where model is saved at
        n_iter -- amount of times data is trained with
        train_data -- training data in BILOU Format

        Returns:
        output_dir -- path to model
        """
        dropout = decaying(0.6 , 0.2, 1e-4)
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
        disabled = nlp.disable_pipes(*other_pipes)
        logging.info("Started training entities...")
        optimizer = nlp.begin_training()
        for iteration in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations, _ = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=next(dropout),  # dropout - make it harder to memorise data
                    sgd=optimizer,
                    losses=losses,
                )            
            p, r, f = self.evaluate_entity(nlp)
            self.entity_score.append([p, r, f])
            logging.info("Finished %s iteration for NER with %s losses", iteration, losses)
            self.losses_ner.append(losses)
        logging.info("Finished training entities...")
        disabled.restore()

        # save model to output directory
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.to_disk(output_dir)
            logging.info("Saved entity model to %s", output_dir)

        return output_dir

    def train_intent(self, nlp, output_dir, train_data,n_iter, dropout):
        """Load the model, set up the pipeline and train the entity recognizer.
        

        Keyword arguments:
        model -- path to the model if existent
        output_dir -- path where model is saved at
        n_iter -- amount of times data is trained with
        train_data -- training data in BILOU Format

        Returns:
        output_dir -- path to model
        """
        dropout = decaying(0.6 , 0.2, 1e-4)
        pipe_exceptions = ["trf_textcat", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
        disabled = nlp.disable_pipes(*other_pipes)
        logging.info("Started training intents...")
        optimizer = nlp.resume_training()
        optimizer.alpha = 0.001
        optimizer.trf_weight_decay = 0.005
        optimizer.L2 = 0.0

        learn_rate=2e-5
        batch_size=8
        learn_rates = cyclic_triangular_rate(
            learn_rate / 3, learn_rate * 3, 2 * len(train_data) // batch_size
        )
        for iteration in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                optimizer.trf_lr = next(learn_rates)
                texts, _, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    sgd=optimizer,
                    drop=next(dropout),  # dropout - make it harder to memorise data
                    losses=losses
                )
            self.losses_cat.append(losses)
            p, r, f = self.evaluate_intent(nlp)
            self.cat_scores.append([p, r, f])
            logging.info("Finished %s iteration for text classification with %s losses", iteration, losses)
            #if cat_score <= self.cat_scores[-2]:
                #break
        logging.info("Finished training intents...")
        disabled.restore()

        # save model to output directory
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.to_disk(output_dir)
            logging.info("Saved model to %s", output_dir)

        return output_dir

    def evaluate_entity(self, nlp):
        """evaluates a given model with spacy gold
        

        Keyword arguments:
        model -- path to the trained model
        examples -- path to test data to compare model with

        Return:
        scorer.ents_p -- precision score
        scorer.ents_r -- recall score
        scorer.ents_f -- f1 score
        
        """
        logging.info("Started evaluating entities...")
        examples = self.test_data        
        start_time = time.perf_counter()
        scorer = Scorer()
        for text, annotations, _ in examples:
            doc_gold = nlp.make_doc(text)
            entity = annotations.get("entities")
            gold = GoldParse(doc_gold, entities=entity)
            scorer.score(nlp(text), gold)

        logging.debug("Testing data took %d seconds to run", time.perf_counter() - start_time)
        logging.debug("Score for entity is: p:%f,r:%f,f:%f", scorer.ents_p, scorer.ents_r, scorer.ents_f)
        logging.info("Finished evaluating entities")
        return scorer.ents_p, scorer.ents_r, scorer.ents_f

    def evaluate_intent(self, nlp):
        """evaluates a given model with roc
        

        Keyword arguments:
        model -- path to the trained model
        examples -- path to test data to compare model with

        Return:
        scorer.ents_p -- precision score
        scorer.ents_r -- recall score
        scorer.ents_f -- f1 score
        
        """
        logging.info("Started evaluating intent...")
        start_time = time.perf_counter()
        tp = 1e-8  # True positives
        fp = 1e-8  # False positives
        fn = 1e-8  # False negatives
        tn = 1e-8  # True negatives

        for text, _, annotations in self.test_data:
            doc = nlp(text)
            gold = annotations.get("cats")

            for label, score in doc.cats.items():
                if label not in gold:
                    continue
                if score >= 0.5 and gold[label] >= 0.5:
                    tp += 1.0
                elif score >= 0.5 and gold[label] < 0.5:
                    fp += 1.0
                elif score < 0.5 and gold[label] < 0.5:
                    tn += 1
                elif score < 0.5 and gold[label] >= 0.5:
                    fn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = 2 * (precision * recall) / (precision + recall)

        logging.debug("Testing data took %f seconds to run", time.perf_counter() - start_time)
        logging.debug("Score for intent is %f, %f, %f", precision, recall, f_score)
        logging.info("Finished evaluating intent")
        
        return precision, recall, f_score

    def balance_data(self, data, data_type, label_threshold = 0.4):
        """ Adjust dataset if it has more than defined threshold of a label by deleting samples until threshold is met

        Args:
            data (dict): Labelled training data
            data_type (string): Which kind of dataset dependent on the dataset
            label_threshold (float, optional): Maximum percentage a  label can occur in dataset. Defaults to 0.4.

        Returns:
            [dict]: More evenly adjusted dataset
        """

        all_intents = []
        examples = data[data_type]

        [all_intents.append(example["intent"]) for example in examples]
        cat_count = dict.fromkeys(set(all_intents), 0)

        for cat in cat_count.keys():
            cat_count[cat] = all_intents.count(cat)
        
        sub_datasets = defaultdict(list)

        for item in examples:
            sub_datasets[item['intent']].append(item)

        cat_count = dict(sorted(cat_count.items(), key=lambda item: item[1], reverse=True))
        intent_length = len(all_intents)

        for cat in cat_count.keys():
            if all_intents.count(cat) > (label_threshold) * intent_length:
                rest_intent_len = intent_length - all_intents.count(cat)
                cat_count[cat] = int(rest_intent_len * label_threshold)
                temp_list = sub_datasets[cat]
                random.shuffle(temp_list)
                sub_datasets[cat] = temp_list[:cat_count[cat]]
                intent_length = intent_length + cat_count[cat] - all_intents.count(cat)
                
        data[data_type] = list(itertools.chain.from_iterable(sub_datasets.values()))

        return data
        

    def convert_data(self, data, data_type):
        """ takes training Data in meta engine Format and converts it into the BILOU Format

        Args:
            data (dict): the dataset itself in type dict
            data_type(str): Type of dataset, either "trainingSet", "validationSet" or "testSet"

        Returns:
            BILOU: Training data in the bilou format
        """

        training_data = []

        examples = data[data_type]

        all_intents = []
        [all_intents.append(example["intent"]) for example in examples]
        dict_cat = dict.fromkeys(set(all_intents), 0)

        for cat in dict_cat.keys():
            dict_cat[cat] = all_intents.count(cat)
        logging.debug("There are this many intents: %s", dict_cat)         

        for example in examples:
            intents = {}
            entities = []
            sentence = example["sample"]
            cat = example["intent"]
            for cats in dict_cat.keys():
                if cats == cat:
                    intents[cats] = 1.0
                else:
                    intents[cats] = 0.0 
            for entity in example["entities"]:
                label = entity["categories"][0]  # TODO clear up if there can be multiple entity categories for one entity
                label_start = entity["token"]["start"]
                label_end = entity["token"]["end"]
                tupel = (label_start, label_end, label)
                entities.append(tupel)

            # puts the labels in the right format into TRAINING_DATA

            training_data = training_data + [(sentence, {"entities": entities}, {'cats': intents})]

        return training_data

    def create(self, model, output_dir):
        if model is not None:
            nlp = spacy.load(model)
            logging.info("Loaded model %s", model)
        else:
            nlp = spacy.blank("de")  # create blank Language class
            logging.info("Created blank 'de' model")


        # save model to output directory
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.to_disk(output_dir)
            logging.info("Saved model to %s", output_dir)

    def unpack_hyperparam(self, hyperparameters):
        """Unpacks the values of the hyperparam and returns as variables

        Args:
            hyperparam ([dict]): Has the value for each hyperparameter 

        Returns:
            [float]: Amount of iteration for entity training
            [float]: Droprate for entity training
            [float]: Amount of iteration for entity training
            [float]: Droprate for intent training

        """
        if hyperparameters is not None:
            if self.hyperparameters.keys() != hyperparameters.keys():
                logging.warning("The received hyperparameters don't exist, using default values...")
                ent_itr, ent_drop, cat_itr, cat_drop = self.hyperparameters.values()
            else:
                 ent_itr, ent_drop, cat_itr, cat_drop = hyperparameters.values()               
        else:
            ent_itr, ent_drop, cat_itr, cat_drop = self.hyperparameters.values()
        
        return ent_itr, ent_drop, cat_itr, cat_drop

    def update(self, dataset, model, output_dir, hyperparameters = None):
        """Trains/Updates a language model

        Args:
            dataset (dict): labelled training_data
            model (string): location of the updated model, set None if training from scratch
            output_dir (string): location where new model should be saved
            hyperparameters (dict, optional): Includes values to adjust hyperparameters for training. Defaults to None then uses default values.

        Returns:
            [dict]: Score of model, components it supports, hyperparameter that were used
        """

        logging.info("Setting hyperparameters...")
        ent_itr, ent_drop, cat_itr, cat_drop = self.unpack_hyperparam(hyperparameters)

        logging.info("Start spacy model training for model %s", model)
        is_using_gpu = spacy.prefer_gpu()
        if is_using_gpu:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        logging.info("Spacy is using GPU: %s",is_using_gpu)
        warnings.filterwarnings("once", category=UserWarning, module='spacy')  # ignore repeating misalgminent errors

        # dataset = self.balance_data(dataset,"trainingSet") Comment in if dataset is imbalanced
        training_data = self.convert_data(dataset, "trainingSet")
        self.test_data = self.convert_data(dataset, "testSet")

        nlp = self.load_model(model, training_data)
        self.train_entity(nlp, output_dir, training_data, ent_itr, ent_drop)
        model = self.train_intent(nlp, output_dir, training_data, cat_itr, cat_drop)
        logging.info("Finished spacy model training for model at %s", model)

        logging.info("All losses for each iteration in entity %s", self.losses_ner)
        logging.info("All losses for each iteration in intent %s", self.losses_cat)
        logging.info("All scores for each iteration in entity %s", self.entity_score)
        logging.info("All scores for each iteration in intent %s", self.cat_scores)

        p, r, f = self.entity_score[-1]
        intent_score = self.cat_scores[-1]

        self.scores = {"Precision": p, "Recall": r, "F1-Score": f, "IntentScore": intent_score}

        return {"Scores": self.scores, "Components": self.components, "Hyperparameters": self.hyperparameters}

if __name__ == "__main__":

    """testing purposes only"""

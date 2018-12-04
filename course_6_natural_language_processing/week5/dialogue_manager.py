import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import pairwise_distances_argmin
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from utils import unpickle_file
from utils import load_embeddings
from utils import question_to_vec
from utils import text_prepare


class ThreadRanker(object):
    """
    Class to rank the different StackOverflow threads
    """

    def __init__(self, paths):
        """
        Constructor for the ThreadRanker, will set the word_embeddings,
        embeddings_dim and thread_embeddings_dir

        Parameters
        ----------
        paths : dict
            Where the keys are names, and the values are lists of paths
            relative to this directory
        """
        self.word_embeddings, self.embeddings_dim = \
            load_embeddings(Path(*paths['WORD_EMBEDDINGS']))
        self.thread_embeddings_dir = Path(*paths['THREAD_EMBEDDINGS_FOLDER'])

    def __load_embeddings_by_tag(self, tag_name):
        """
        Returns the thread_id and the thread_embeddings

        Parameters
        ----------
        tag_name : str
            Name of the tag

        Returns
        -------
        thread_ids : array-like, shape (n_ids, )
            Array of the ids corresponding to the tag
        thread_embeddings : np.array, shape (n_ids, embedding_dim)
            The embedding to the corresponding tag
        """

        embeddings_path = self.thread_embeddings_dir.joinpath(tag_name + '.pkl')
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)

        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """
        Returns id of the most similar thread for the question.

        The search is performed across the threads with a given tag.

        Parameters
        ----------
        question : str
            The question asked
        tag_name : str
            The tag for the question

        Returns
        -------
        int
            The id of the most similar thread of the question
        """

        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        question_vec = question_to_vec(question=question,
                                       embeddings=self.word_embeddings,
                                       dim=thread_embeddings.shape[1])

        best_thread = pairwise_distances_argmin(question_vec[np.newaxis, ...],
                                                thread_embeddings,
                                                metric='cosine')
        
        return thread_ids[best_thread][0]


class DialogueManager(object):
    """
    Class for the dialogue manager
    """

    def __init__(self, paths):
        """
        Constructor for the DialogueManager

        - Loads the intent recognizer (is this about programming, or just
          chit-chatting?)
        - Loads the tf-idf vectorizer (the vectorizer trained on the dialogue
          and StackOverflow thread questions)

        Parameters
        ----------
        paths : dict
            Where the keys are names, and the values are lists of paths
            relative to this directory
        """
        print("Loading resources...")

        # Create the chatbot
        self.create_chitchat_bot()

        # Intent recognition:
        self.intent_recognizer = \
            unpickle_file(Path(*paths['INTENT_RECOGNIZER']))
        self.tfidf_vectorizer = \
            unpickle_file(Path(*paths['TFIDF_VECTORIZER']))

        self.ANSWER_TEMPLATE = ('I think its about {}\n'
                                'This thread might help you: '
                                'https://stackoverflow.com/questions/{}')

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(Path(*paths['TAG_CLASSIFIER']))
        self.thread_ranker = ThreadRanker(paths)

    def create_chitchat_bot(self):
        """
        Initializes self.chitchat_bot with some conversational model.
        """

        self.chatbot = \
            ChatBot('MrStack2000Bot',
                    trainer='chatterbot.trainers.ChatterBotCorpusTrainer')

        # Train on the english corpus
        self.chatbot.train("chatterbot.corpus.english")

        # Train for own conversation
        self.chatbot.set_trainer(ListTrainer)
        self.chatbot.train(
            ['What is PEP20?',
             'The zen of python',
             'Where can I learn about it?',
             'On the interwebz',
             'Where on the interwebz?'
             'https://www.python.org/dev/peps/pep-0020/']
        )
        self.chatbot.train([
            'What is AI?',
            ('I like the definition: AI is what we think is intelligent, but'
             'which has not yet been achieved')
        ])

    def generate_answer(self, question):
        """
        Combines StackOverflow and chitchat parts using intent recognition.

        Parameters
        ----------
        question : str
            The question asked

        Returns
        -------
        str
            The answer
        """

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the
        # question.

        prepared_question = text_prepare(question)

        if prepared_question == '':
            # Text preparation is an empty string, tf_idf won't work
            return self.chatbot.get_response(question)

        features = self.tfidf_vectorizer.transform(prepared_question.split())
        intent = self.intent_recognizer.predict(features)[0]

        # Chit-chat part:
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chatbot.get_response(question)
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            # NOTE: [0] as we are just after the first tag
            tag = self.tag_classifier.predict(features)[0]
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(question, tag)
           
            return self.ANSWER_TEMPLATE.format(tag, thread_id)

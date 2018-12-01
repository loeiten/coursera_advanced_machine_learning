from sklearn.metrics.pairwise import pairwise_distances_argmin
from chatterbot import ChatBot
from utils import unpickle_file
from utils import load_embeddings
from utils import question_to_vec


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
            load_embeddings(*paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_dir = paths['THREAD_EMBEDDINGS_FOLDER']

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

        embeddings_path = self.thread_embeddings_dir.join(tag_name + '.pkl')
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
                                       embeddings=thread_embeddings,
                                       dim=thread_embeddings.shape[1])

        best_thread = pairwise_distances_argmin(question_vec,
                                                thread_embeddings)
        
        return thread_ids[best_thread]


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

        # Declarations
        self.chatbot = None

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = ('I think its about {}\n'
                                'This thread might help you: '
                                'https://stackoverflow.com/questions/{}')

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)

    def create_chitchat_bot(self):
        """
        Initializes self.chitchat_bot with some conversational model.
        """

        self.chatbot = ChatBot('MrStack2000Bot',
                          trainer='chatterbot.trainers.ChatterBotCorpusTrainer')

        self.chatbot.train("chatterbot.corpus.english")

       
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
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question = #### YOUR CODE HERE ####
        features = #### YOUR CODE HERE ####
        intent = #### YOUR CODE HERE ####

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = #### YOUR CODE HERE ####
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = #### YOUR CODE HERE ####
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = #### YOUR CODE HERE ####
           
            return self.ANSWER_TEMPLATE.format(tag, thread_id)

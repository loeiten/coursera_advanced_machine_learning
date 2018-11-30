#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import requests
import time
import argparse
import os
import json
import sys
from requests.compat import urljoin


if 'UTF-8' not in sys.stdout.encoding:
    msg = ("Terminal can't handle UTF-8\n\n"
           "Try exporting:\n"
           "export LC_ALL=en_US.UTF-8\n"
           "export LANG=en_US.UTF-8\n"
           "export LANGUAGE=en_US.UTF-8\n")
    raise RuntimeError(msg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default='')
    return parser.parse_args()


def is_unicode(text):
    return len(text) == len(text.encode())


class BotHandler(object):
    """
    BotHandler is a class which implements all back-end of the bot.

    Methods
    -------
    get_updates(offset=None, timeout=30)
        Checks for new messages
    send_message(chat_id, text)
        Posts new message to user
    get_answer(question)
        Computes the most relevant on a user's question
    """

    def __init__(self, token, dialogue_manager):
        """
        Constructor for the bot handler

        Sets token and dialogue_manager

        Parameters
        ----------
        token : str
            The token for the Rest API
        dialogue_manager : DialogueManager-like
            The dialogue-manager
            Must contain the method `generate_answer`
        """
        self.token = token
        self.api_url = 'https://api.telegram.org/bot{}/'.format(token)
        self.dialogue_manager = dialogue_manager

    def get_updates(self, offset=None, timeout=30):
        """
        Fetches updates from the bot

        Parameters
        ----------
        offset : None or int
            Conversation offset, see
            https://core.telegram.org/bots/api#getupdates
            for details
        timeout : int
            Timeout for long pooling

        Returns
        -------
        list
            The result of the update
        """
        params = {'timeout': timeout, 'offset': offset}
        raw_resp = requests.get(urljoin(self.api_url, 'getUpdates'), params)
        try:
            resp = raw_resp.json()
        except json.decoder.JSONDecodeError as e:
            print('Failed to parse response {}: {}.'.
                  format(raw_resp.content, e))
            return []

        if 'result' not in resp:
            return []
        return resp['result']

    def send_message(self, chat_id, text):
        """
        Send a message to the bot

        Parameter
        ---------
        chat_id : int
            The chat id to send to
        text : str
            The message to send to the bot

        Returns
        -------
        Response
            The response of the post action
        """
        params = {'chat_id': chat_id, 'text': text}
        return requests.post(urljoin(self.api_url, 'sendMessage'), params)

    def get_answer(self, question):
        """
        Get the answer to the question

        Parameters
        ----------
        question : str
            The question as a string

        Returns
        -------
        str
            The answer from the dialogue manager
        """
        if question == '/start':
            return 'Hi, I am your project bot. How can I help you today?'
        return self.dialogue_manager.generate_answer(question)


class SimpleDialogueManager(object):
    """
    This is the simplest dialogue manager to test the telegram bot.
    Your task is to create a more advanced one in dialogue_manager.py."
    """

    @staticmethod
    def generate_answer(_):
        """
        Replies "Hello, world!" irrespective of the input

        Parameters
        ----------
        _ : str
            The input string

        Returns
        -------
        str
            The "Hello, world!" string
        """
        return "Hello, world!"


def main():
    """
    The main function of the conversation service
    """
    args = parse_args()
    token = args.token

    if not token:
        if 'TELEGRAM_TOKEN' not in os.environ:
            print('Please, set bot token through '
                  '--token or TELEGRAM_TOKEN env variable')
            return
        token = os.environ['TELEGRAM_TOKEN']

    #################################################################

    # Your task is to complete dialogue_manager.py and use your
    # advanced DialogueManager instead of SimpleDialogueManager.

    # This is the point where you plug it into the Telegram bot.
    # Do not forget to import all needed dependencies when you do so.

    simple_manager = SimpleDialogueManager()
    bot = BotHandler(token, simple_manager)

    ###############################################################

    print('Ready to talk!')
    offset = 0
    while True:
        updates = bot.get_updates(offset=offset)
        for update in updates:
            print('An update received.')
            if 'message' in update:
                chat_id = update['message']['chat']['id']
                if 'text' in update['message']:
                    text = update['message']['text']
                    if is_unicode(text):
                        print('Update content: {}'.format(update))
                        bot.send_message(chat_id,
                                         bot.get_answer(
                                             update['message']['text']))
                    else:
                        msg = ('Hmm, you are sending some weird characters to '
                               'me...')
                        bot.send_message(chat_id, msg)
            offset = max(offset, update['update_id'] + 1)
        time.sleep(1)


if __name__ == '__main__':
    main()



from encoder import *
from decoder import *

class Model:
    def __init__(self, encoder, decoder, settings):
        self.encoder = encoder
        self.decoder = decoder
        self.settings = settings

    def train(self):
        pass

    def predict(self):
        pass

    def loss(self):
        pass


from neuroboa.layers.layer import Layer

class Word2VecEmbedding(Layer):
    def __init__(self, input_vocab, embed_dim, initializer = "", *args, **kwargs):
        super(Word2VecEmbedding, self).__init__(*args, **kwargs)
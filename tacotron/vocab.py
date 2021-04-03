import string

VOCAB = [" "] + list(string.ascii_lowercase)


class CharVocab(object):
    def __init__(self, vocab=VOCAB):
        vocab = ["<p>", "<unk>"] + vocab

        self.sym2id = {sym: id for id, sym in enumerate(vocab)}
        self.id2sym = {id: sym for id, sym in enumerate(vocab)}

    def __len__(self):
        return len(self.sym2id)

    def encode(self, syms):
        return [self.sym2id[sym] if sym in self.sym2id else self.sym2id["<unk>"] for sym in syms]

    def decode(self, ids):
        return "".join(self.id2sym[id] for id in ids)

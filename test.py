from models.mlp import mlp
from models.rnn import rnn

if __name__ == '__main__':
    # obj = mlp()
    # obj.model()
    # obj.train()

    obj = rnn(cellType="LSTM", wrapDropout=True, redChars=True)
    obj.model()
    obj.train()

import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import pandas as pd
from torchtext.data import Iterator, BucketIterator
from torchtext.data import Field
from torchtext.data import TabularDataset

tokenize = lambda x: x.split()
TEXT = Field(sequential=True, tokenize=tokenize, lower=True)  # tokenizer
LABEL = Field(sequential=False, use_vocab=False)  # sequential=False, use_vocab=False already numericalized

print(pd.read_csv("kaggle_data/train.csv").head(2))

tv_datafields = [("id", None),  # we won't be needing the id, so we pass in None as the field
                 ("comment_text", TEXT), ("toxic", LABEL),
                 ("severe_toxic", LABEL), ("threat", LABEL),
                 ("obscene", LABEL), ("insult", LABEL),
                 ("identity_hate", LABEL)]
trn, vld = TabularDataset.splits(
    path="kaggle_data",  # the root directory where the kaggle_data lies
    train='train.csv', validation="valid.csv",
    format='csv',
    skip_header=True,
    # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as kaggle_data!
    fields=tv_datafields)

tst_datafields = [("id", None),  # we won't be needing the id, so we pass in None as the field
                  ("comment_text", TEXT)]
tst = TabularDataset(
    path="kaggle_data/test.csv",  # the file path
    format='csv',
    skip_header=True,
    # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as kaggle_data!
    fields=tst_datafields)

TEXT.build_vocab(trn)
TEXT.vocab.freqs.most_common(10)

train_iter, val_iter = BucketIterator.splits(
    (trn, vld),  # we pass in the datasets we want the iterator to draw kaggle_data from
    batch_sizes=(64, 64),
    device=-1,  # if you want to use the GPU, specify the GPU number here
    sort_key=lambda x: len(x.comment_text),
    # the BucketIterator needs to be told what function it should use to group the kaggle_data.
    sort_within_batch=False,
    repeat=False  # we pass repeat=False because we want to wrap this Iterator layer.
)

batch = next(train_iter.__iter__())
batch.__dict__.keys()

test_iter = Iterator(tst, batch_size=64, device=-1, sort=False, sort_within_batch=False, repeat=False)


class BatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars  # we pass in the list of attributes for x and y

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)  # we assume only one input in this wrapper

            if self.y_vars is not None:  # we will concatenate y into a single tensor
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).float()
            else:
                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)


train_dl = BatchWrapper(train_iter, "comment_text",
                        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
valid_dl = BatchWrapper(val_iter, "comment_text",
                        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
test_dl = BatchWrapper(test_iter, "comment_text", None)

next(train_dl.__iter__())


class SimpleBiLSTMBaseline(nn.Module):
    def __init__(self, hidden_dim, emb_dim=300,
                 spatial_dropout=0.05, recurrent_dropout=0.1, num_linear=1):
        super().__init__()  # don't forget to call this!
        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1, dropout=recurrent_dropout)
        self.linear_layers = []
        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim, 6)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
        preds = self.predictor(feature)
        return preds


em_sz = 100
nh = 500
nl = 3
model = SimpleBiLSTMBaseline(nh, emb_dim=em_sz)

opt = optim.Adam(model.parameters(), lr=1e-2)
loss_func = nn.BCEWithLogitsLoss()
epochs = 4

for epoch in range(1, epochs + 1):
    running_loss = 0.0
    running_corrects = 0
    model.train()  # turn on training mode
    for x, y in tqdm.tqdm(train_dl):  # thanks to our wrapper, we can intuitively iterate over our kaggle_data!
        opt.zero_grad()

        preds = model(x)
        loss = loss_func(preds, y)
        loss.backward()
        opt.step()

        # running_loss += loss.kaggle_data[0] * x.size(0)
        running_loss += loss.data.item() * x.size(0)

    epoch_loss = running_loss / len(trn)

    # calculate the validation loss for this epoch
    val_loss = 0.0
    model.eval()  # turn on evaluation mode
    for x, y in valid_dl:
        preds = model(x)
        loss = loss_func(preds, y)
        # val_loss += loss.kaggle_data[0] * x.size(0)
        val_loss += loss.data.item() * x.size(0)

    val_loss /= len(vld)
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))

test_preds = []
for x, y in tqdm.tqdm(test_dl):
    preds = model(x)
    # if you're kaggle_data is on the GPU, you need to move the kaggle_data back to the cpu
    # preds = preds.kaggle_data.cpu().numpy()
    preds = preds.data.numpy()
    # the actual outputs of the model are logits, so we need to pass these values to the sigmoid function
    preds = 1 / (1 + np.exp(-preds))
    test_preds.append(preds)
test_preds = np.hstack(test_preds)

print(test_preds)

df = pd.read_csv("kaggle_data/test.csv")
for i, col in enumerate(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]):
    df[col] = test_preds[:, i]
print(df.head(3))

import torch
import torch.nn.functional as F
import DataSet
import numpy as np
from sklearn.metrics import accuracy_score

from model.TextCNN import TextCNN
from model.TextRCNN import TextRCNN
from model.TextRNN import TextRNN
from model.TextRNN_Attention import TextRNN_Attention
from model.Transformer import Transformer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model(test_iter, model, device):
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    accuracy = 0
    y_true = []
    y_pred = []
    total_test_num = len(test_iter.dataset)
    for batch in test_iter:
        feature = batch.text
        target = batch.label
        with torch.no_grad():
            feature = torch.t(feature)
        feature, target = feature.to(device), target.to(device)
        out = model(feature)
        loss = F.cross_entropy(out, target)
        total_loss += loss.item()
        accuracy += (torch.argmax(out, dim=1)==target).sum().item()
        y_true.extend(target.cpu().numpy())
        y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
    print('>>> Test loss:{}, Accuracy:{} \n'.format(total_loss/total_test_num, accuracy/total_test_num))
    score = accuracy_score(y_true, y_pred)
    print(score)
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_true, y_pred)
    print(confusion_matrix)
    from sklearn.metrics import classification_report
    target_names = ['差评', '中评', '好评']
    print(classification_report(y_true, y_pred, target_names=target_names))

def train_model(train_iter, dev_iter, model, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    epochs = 10
    print('training...')
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        accuracy = 0
        total_train_num = len(train_iter.dataset)
        for batch in train_iter:
            feature = batch.text
            target = batch.label
            with torch.no_grad():
                feature = torch.t(feature)
            feature, target = feature.to(device), target.to(device)
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            accuracy += (torch.argmax(logit, dim=1)==target).sum().item()
        print('>>> Epoch_{}, Train loss is {}, Accuracy:{} \n'.format(epoch,loss.item()/total_train_num, accuracy/total_train_num))
        model.eval()
        total_loss = 0.0
        accuracy = 0
        total_valid_num = len(dev_iter.dataset)
        for batch in dev_iter:
            feature = batch.text  # (W,N) (N)
            target = batch.label
            with torch.no_grad():
                feature = torch.t(feature)
            feature, target = feature.to(device), target.to(device)
            out = model(feature)
            loss = F.cross_entropy(out, target)
            total_loss += loss.item()
            accuracy += (torch.argmax(out, dim=1)==target).sum().item()
        print('>>> Epoch_{}, Valid loss:{}, Accuracy:{} \n'.format(epoch, total_loss/total_valid_num, accuracy/total_valid_num))

def saveModel(model,name):
    torch.save(model, 'done_model/'+name+'_model.pkl')

model = Transformer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter, val_iter, test_iter = DataSet.getIter()

if __name__ == '__main__':
    train_model(train_iter, val_iter, model, device)
    saveModel(model,'transformer')
    test_model(test_iter, model, device)
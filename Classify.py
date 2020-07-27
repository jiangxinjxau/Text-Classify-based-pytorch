import torch
import DataSet
from torchtext import data

def getModel(name):
    model = torch.load('done_model/'+name+'_model.pkl')
    return model

model = getModel('textrcnn')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
sent1 = '垃圾，这个东西最好别买'
demo = [data.Example.fromlist(data=[sent1,0],fields=[('text',DataSet.getTEXT()),('label',DataSet.getLabel())])]
demo_iter = data.BucketIterator(dataset=data.Dataset(demo,[('text',DataSet.getTEXT()),('label',DataSet.getLabel())]), batch_size=256, shuffle=True,sort_key=lambda x:len(x.text), sort_within_batch=False, repeat=False)
for batch in demo_iter:
    feature = batch.text
    target = batch.label
    with torch.no_grad():
        feature = torch.t(feature)
    feature, target = feature.to(device), target.to(device)
    out = model(feature)
    if torch.argmax(out, dim=1).item() == 0:
        print('差评')
    elif torch.argmax(out, dim=1).item() == 2:
        print('好评')
    else:
        print('中评')
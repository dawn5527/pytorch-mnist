import numpy
from flask import Flask, jsonify, render_template, request
from PIL import Image

import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

import model
# webapp
app = Flask(__name__)

def predict_with_pretrain_model(sample,model):
    #fix dataset
    sample = -sample + 255
    img = Image.fromarray(sample)
    transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
    sample = transform(img).float()
    sample = Variable(sample, volatile=True)
    sample = sample.unsqueeze(0)
    
    model = model.Net()
    model.load_state_dict(torch.load('./results/lenet.pkl'))
    out = model.predict(sample)
    
    return out.data[0].tolist()

@app.route('/api/mnist', methods=['POST'])
def mnist():
    
    input = ((numpy.array(request.json, dtype=numpy.uint8))).reshape(28, 28)
    output = predict_with_pretrain_model(input,model)
    return jsonify(results=output)


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

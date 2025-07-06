
import random
from micrograd.engine import Value


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] # no of inputs
        self.b = Value(random.uniform(-1, 1))  # bias 

    def __call__(self, x):

        # w * x + b
        # converts xi to Value
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):

        # returns list of parameters
        return self.w + [self.b]
        

class Layer:

    def __init__(self, nin, nout):
        # nout : no of ouputs this layer has = # neurons
        # nin : every neuron take nin inputs
        
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
            

class MLP:

    def __init__(self, nin, nouts):

        #nouts : [2, 3, 4]: size of the layers (including output)
        #nin : size of input
        
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):

        for layer in self.layers:
            # x : outputs after each layer
            x = layer(x)

        return x  #final output

    def parameters(self):
        # gives the parameters that have to be updated
        return [p for layer in self.layers for p in layer.parameters()]

        
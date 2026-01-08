import torch

class EMA:
    def __init__(self, model, beta):
        self.model = model
        self.step = 0
        self.beta = beta
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
        self.backup = {}

    def update(self):
        self.step += 1
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.beta) * param.data + self.beta * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
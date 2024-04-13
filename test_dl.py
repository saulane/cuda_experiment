import numpy as np
import torch
import torch.nn as nn

model = nn.Sequential(
          nn.Linear(3,2, bias=True),
        )

if __name__ == "__main__":
    loss = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(),  lr=0.001)
    inputs = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([2.0])

    weights = torch.tensor([[0.1, 0.2, 0.3]])
    bias = torch.tensor([0.1])
    # layer = torch.nn.Linear(3,2, bias=True)

    model[0].weight.data = weights
    model[0].bias.data = bias

    for i in range(10):
        print(f"-----------------EPOCH {i}-----------------")
        optimizer.zero_grad()
        outputs = model(inputs)
    # print(outputs)

        l = loss(outputs, target)
        print(f"Loss epoch {i}: {l}")
        l.backward()
        optimizer.step()
        print("Weights/Bias:",model[0].weight.data, model[0].bias.data)

        print(model[0].weight.grad)
        print(model[0].bias.grad)


# -----------------EPOCH 9-----------------
# Loss epoch 9: 0.14448770880699158
# Weights/Bias: tensor([[0.1088, 0.2175, 0.3263]]) tensor([0.1088])
# tensor([[-0.7602, -1.5205, -2.2807]])
# tensor([-0.7602])


# -----------------EPOCH 9-----------------
# Loss epoch 9: 0.144488
# Weights / bias:  0.108753 0.217505 0.326258/ 0.108753
# Weight Gradient:
# [-0.760231] [-1.520462] [-2.280693] 
# Bias Gradient:
# [-0.760231]
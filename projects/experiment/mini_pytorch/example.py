import torch

model = torch.jit.load("temporal_wnet_small_inputema_traced.pt")
model.eval()

x = torch.rand([2, 6, 200, 200]).to("cuda:0")
r = torch.zeros([2, 6, 200, 200]).to("cuda:0")
x, r = model.forward(x, r)

print(x)
print(r)

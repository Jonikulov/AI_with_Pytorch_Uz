import torch

# Train Data (o'rgatishdagi ma'lumotlar)
x_soat = [1.0, 2.0, 3.0]
y_baho = [2.0, 4.0, 6.0]

w = torch.tensor([1.0], requires_grad=True)  # taxminiy qiymat

def forward(x):
    """To'g'riga hisoblash funksiyasi"""
    return x * w

def loss(y_true, y_pred):
    """Xatolik (loss)ni hisoblash funksiyasi"""
    return (y_true - y_pred) ** 2

print(f"Bashorat (training'dan avval): Soat: 4, Baho: {forward(4).item()}\n")

# Training loop
epochs_num = 10
lr = 0.01  # learning rate
for epoch in range(epochs_num):
    for x, y in zip(x_soat, y_baho):
        y_pred = forward(x)  # forward
        l = loss(y, y_pred)  # loss
        l.backward()  # calculate backward
        print(f'\tx: {x}\ty: {y}\tgrad: {w.grad.item():.3f}')
        w.data = w.data - lr * w.grad.item()  # update the weight
        w.grad.data.zero_()  # zero the gradient for the next epoch
    print(f'Epoch {epoch+1}/{epochs_num} | Loss: {l.item():.3f}\n')

print(f"Bashorat (training'dan keyin): Soat: 4, Baho: {forward(4).item()}")

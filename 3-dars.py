# Train Data (o'rgatishdagi ma'lumotlar)
x_soat = [1.0, 2.0, 3.0]
y_baho = [2.0, 4.0, 6.0]

w = 1.0  # weight uchun dastlabki taxminiy qiymat

def forward(x):
    """To'g'riga hisoblash funksiyasi"""
    return x * w

def loss(y_true, y_pred):
    """Xatolik (loss)ni hisoblash funksiyasi"""
    return (y_true - y_pred) ** 2

def gradeint(x, y):  # d_loss / d_weight
    """Gradientni hisoblash funksiyasi"""
    return 2 * x * (x * w - y)

print(f"Bashorat (training'dan avval): Soat: 4, Baho: {forward(4)}\n")

# Training loop
epochs_num = 10
lr = 0.01  # learning rate
for epoch in range(epochs_num):
    print(f'Epoch {epoch+1}/{epochs_num}:')
    for x, y in zip(x_soat, y_baho):
        y_pred = forward(x)  # forward
        error = loss(y, y_pred)  # loss
        grad = gradeint(x, y)  # gradient
        w = w - lr * grad  # update the weight

        print(f'\tx: {x}\ty: {y}\tpred: {y_pred:.3f}\t' \
              f'loss: {error:.3f}\tgrad: {grad:.3f}')

print(f"\nBashorat (training'dan keyin): Soat: 4, Baho: {forward(4)}")

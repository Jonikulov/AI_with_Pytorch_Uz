import numpy as np
import matplotlib.pyplot as plt

# data
x_soat = np.array([1, 2, 3])
y_baho = np.array([2, 4, 6])

def forward(x, w):
    """To'g'riga hisoblash funksiyasi"""
    return x * w

def loss(y_true, y_pred):
    """Xatolik (loss)ni hisoblash funksiyasi"""
    return (y_pred - y_true)**2

# Grafik uchun Weight va MSE qiymatlari
weights = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    print(f'Weight={w:.2f}')
    print(f'\tSoat\tBaho\tBashorat\tXatolik')
    errors = []

    for x, y in zip(x_soat, y_baho):
        y_pred = forward(x, w)
        error = loss(y, y_pred)
        print(f'\t{x}\t{y}\t{y_pred:.3f}\t\t{error:.3f}')
        errors.append(error)
    
    mse = sum(errors) / len(errors)  # mean squared error
    print(f'MSE={mse}\n')
    weights.append(w)
    mse_list.append(mse)

# Weight va MSE qiymatlari grafikda
plt.plot(weights, mse_list, linewidth=4)
plt.ylabel('Loss', fontsize=20)
plt.xlabel('Weight', fontsize=20)
plt.grid()
plt.show()
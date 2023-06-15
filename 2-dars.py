import numpy as np
import matplotlib.pyplot as plt

x_soat = np.array([1.0, 2.0, 3.0])
y_baho = np.array([2.0, 4.0, 6.0 ])

def forward(x):
    """To'g'riga hisoblash funksiyasi"""
    return x * w

def loss(x, y):
    """Xatolik (Loss) ning funksiyasi"""
    y_pred = forward(x)
    return (y_pred - y)**2

# Grafikni yaratib olishimiz uchun konteynerlar
w_list = []
mse_list = []

# w ni 0 dan 4 gacha oraliqda hisblash 
for w in np.arange(0.0, 4.1, 0.1):
    print(f"w = {w}")
    L_umum=0
    
    for x_hb_qiym, y_hb_qiym in zip(x_soat, y_baho):
        y_hb_bash = forward(x_hb_qiym)
        L_hb_qiym = loss(x_hb_qiym, y_hb_qiym)
        L_umum += L_hb_qiym
        print("\t", "{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(x_hb_qiym, y_hb_qiym, y_hb_bash, L_hb_qiym))
    
    # Har bir ma'lumot uchun MSE ni hisoblaymiz
    print(f"MSE = {L_umum / len(x_soat)}")  # len(x_soat) -> N
    w_list.append(w)
    mse_list.append(L_umum / len(x_soat))

# Grafik natija
plt.plot(w_list, mse_list, linewidth=4)
plt.ylabel('Loss', fontsize=20)
plt.xlabel('W', fontsize=20)
plt.grid()
plt.show()
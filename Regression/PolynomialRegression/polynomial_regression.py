import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
dataset = pd.read_csv("PolynomialRegression/covid19.csv")

print(dataset.info())

dataset = dataset.drop(columns=["ortalama_filyasyon_suresi",
                      "yatak_doluluk_orani",
                      "ortalama_temasli_tespit_suresi",
                      "filyasyon_orani",
                      "ventilator_doluluk_orani",
                      "gunluk_test",
                      "gunluk_iyilesen",
                      "toplam_test",
                      "toplam_hasta",
                      "toplam_vefat",
                      "toplam_iyilesen",
                      "toplam_yogun_bakim",
                      "toplam_entube",
                      "hastalarda_zaturre_oran"])

print(dataset.info())

dataset = dataset.dropna()
dataset = dataset[::-1].reset_index()

print(dataset.info())

X = dataset.loc[:, ["tarih","gunluk_vaka","gunluk_hasta","agir_hasta_sayisi","eriskin_yogun_bakim_doluluk_orani"]]
Y = dataset.loc[:, ["gunluk_vefat"]]

l = len(X.index)
for ind in X.index:
    X["tarih"][ind] = l-ind
    X["gunluk_vaka"][ind] = float(str(X["gunluk_vaka"][ind]).replace(",",""))
    X["gunluk_hasta"][ind] = float(str(X["gunluk_hasta"][ind]).replace(",",""))
    X["agir_hasta_sayisi"][ind] = float(str(X["agir_hasta_sayisi"][ind]).replace(",",""))
    X["eriskin_yogun_bakim_doluluk_orani"][ind] = float(str(X["eriskin_yogun_bakim_doluluk_orani"][ind]).replace(",",""))

print(X)


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree=3)

x_poly = poly_reg.fit_transform(X)

linReg = LinearRegression()

linReg.fit(x_poly,Y)

Y_pred = linReg.predict(x_poly)  
print(Y_pred)
print(len(Y_pred),"forecast for the days")

for i in range(len(Y_pred)):
    print("| Real : "+ str(Y["gunluk_vefat"][i]) + " | Prediction : " + str(Y_pred[i][0]) + " |")

fig2, (bx0,bx1,bx2) = plt.subplots(nrows=3,figsize=(6,10))

bx2.plot(Y, color="blue", label="Real")
bx2.plot(Y_pred, color="red", label="Prediction")
bx2.set_title("Real vs Prediction of Deaths")


grafik_eriskin_yogun_bakim_doluluk_orani = X.loc[:,["eriskin_yogun_bakim_doluluk_orani"]]
bx1.plot(grafik_eriskin_yogun_bakim_doluluk_orani, color="red", label="intensive care")
bx1.set_title("Intensive Care Rate (%)")

grafik_agir_hasta= X.loc[:,["agir_hasta_sayisi"]][::-1]
bx0.plot(grafik_agir_hasta, color="red", label="heavy patient")
bx0.set_title("Heavy Patient Count")


fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(6,10))

ax0.scatter(X.loc[:,["tarih"]],Y, s=10, color="blue", marker="o")
ax0.set_title("Death-Date")

grafik_gunluk_vaka = X.loc[:,["gunluk_vaka"]]
ax1.plot(grafik_gunluk_vaka, color="red", label="daily case")
ax1.set_title("Daily Case Count")

grafik_gunluk_hasta = X.loc[:,["gunluk_hasta"]]
ax2.plot(grafik_gunluk_hasta, color="red", label="daily patient")
ax2.set_title("Daily Patient Count")

plt.legend()
plt.show()


tarih = "03.05.2021"
vakaSayi = 40444
hastaSayi = 2728
agirHastaSayi = 3558
yogunBakimOran = 50

def tarihDonustur(yeniTarih):
    from datetime import date
    dizi = yeniTarih.split(".")
    yeniTarih = date(int(dizi[2]),int(dizi[1]),int(dizi[0]))
    baslangicTarih = date(2021,4,16)
    fark = yeniTarih - baslangicTarih
    return fark.days

fixTarih = tarihDonustur(tarih)

tahmin_edilecek_veri = np.array([fixTarih,vakaSayi,hastaSayi,agirHastaSayi,yogunBakimOran]).reshape(1,-1)

tahmin_edilecek_veri = poly_reg.fit_transform(tahmin_edilecek_veri)

tahmin = linReg.predict(tahmin_edilecek_veri)
print("Prediction for the date of " + tarih + " death is " + str(tahmin))
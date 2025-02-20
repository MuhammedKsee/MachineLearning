def kokal(x):
    return x**(1/2)

def usal(x,level):
    return x**level

def ecludianDistance(A,B):
    if len(A) != len(B):
        return -1
    else:
        len_ = len(A)
        total = 0
        for i in range(len_):
            total += usal(B[i] - int(A[i]),2)
        distance = kokal(total)
        return distance

def most_frequency(dizi):
    counter = 0
    num = dizi[0]
    for i in dizi:
        curr_frequency = dizi.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i
    return num

A = [5,5,"blue"]
B = [10,10,"blue"]
C = [25,25,"red"]
D = [50,50,"blue"]
E = [100,100,"red"]
F = [255,255,"red"]

ornekUzay = [A,B,C,D,E,F]
print("Verinin X değerini girin")
x = int(input())
print("Verinin Y değerini girin")
y = int(input())

ornekNokta = [x,y,""]


def KNNhesapla(uzay,ornek,n):
    tmp_ornekUzay = uzay.copy()
    tmp_ornekNokta = ornek.copy()
    tmp_ornekNokta.pop()

    komsular = []
    boyut = len(ornek)-1
    print("En yakın '"+str(n)+"' Komsu : ")
    for i in range(n):
        enYakinEleman = []
        for j in range(len(tmp_ornekUzay)):
            minMesafe = 1000

            for eleman in tmp_ornekUzay:
                tmp_eleman = eleman.copy()
                tmp_eleman.pop()

                mesafe = ecludianDistance(tmp_eleman,tmp_ornekNokta)

                if mesafe <= minMesafe:
                    minMesafe = mesafe
                    enYakinEleman = eleman.copy()
    print(enYakinEleman)
    renk = enYakinEleman[len(enYakinEleman)-1]
    komsular.append(renk)
    tmp_ornekUzay.remove(enYakinEleman)
    print("komşuların renkleri : ")
    print(komsular)

    return most_frequency(komsular)


ornekNoktaRenk = KNNhesapla(ornekUzay,ornekNokta,n=3)

print("Seçilen Noktanın Rengi :",ornekNoktaRenk)



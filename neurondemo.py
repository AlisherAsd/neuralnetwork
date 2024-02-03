import math
import random


"""  Сама нейросеть, без визуала  """


step = 0.001

def activate(x):
    return 1 / (1 + (1 / (math.e ** x)))

def error(y):
    return y - IDL


""" функция для вычисления локального градиента нейрона
    fii константа
    weightNN вес между слоями 
    innNN слой
"""
def localgrNN(fii, weightNN, innN):
    localgrNN = fii*weightNN*(innN*(1 - innN))
    return localgrNN


""" функция для обучения """

def backpropagation(weightNN, step, localgrNN, activateNN):
    arr = []
    for i in range(len(weightNN)):
        weightNN[i] = weightNN[i] - step*localgrNN*activateNN[i]
        arr.append(weightNN[i])
    return arr


resneu21, resneu22, resneu31, inn1, localgr31= 0, 0, 0, [0, 0], []


weight21 = [0.4, 0.5]
weight22 = [0.3, 0.2]
weight31 = [0.2, 0.3]


for rt in range(1000):
    for q in range(5):

        for w in range(5):

            IDL = (q+w)/10

            inn1[0] = w/5
            inn1[1] = q/5

            resneu21 = 0
            resneu22 = 0

            for i in range(len(inn1)):
                resneu21 += inn1[i] * weight21[i]
                resneu22 += inn1[i] * weight22[i]

            neu21 = activate(resneu21)  
            neu22 = activate(resneu22)

            #print(localgr31)

            inn2 = [neu21, neu22]

            resneu31 = 0

            for i in range(len(inn2)):
                resneu31 += inn2[i] * weight31[i]

            neu31 = activate(resneu31)

            """ константа для формулы локального градиента нейронов (выходящий нейрон) """
            fii = error(neu31)*neu31*(1 - neu31)


            """ локалград для скрытого слоя """
            for i in range(2):
            #    print(fii, weight31[i], inn2[i])
                localgr31.append(localgrNN(fii, weight31[i], inn2[i]))

            weight21bp = backpropagation(weight21, step, localgr31[0], inn1)
            weight22bp = backpropagation(weight22, step, localgr31[1], inn1)
            weight31bp = backpropagation(weight31, step, fii, inn2)

            print(weight21bp,"\n", weight22bp, "\n", weight31bp, "\n result: ", neu31, q, w)


import math
import random
import pygame as pg



"""  Нейронка с pg рандомными весами  """



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


resneu21, resneu22, resneu31, inn1, localgr31= 0, 0, 0, [0, 0, 1], []

step = 0.1
field = [[-1 for x in range(25)] for f in range(25)]

weight21 = [random.random(), random.random(), random.random()]
weight22 = [random.random(), random.random(), random.random()]
weight31 = [random.random(), random.random()]


pg.init()

size = w, h = 500, 500
nn = 20
dps = 0
sc = pg.display.set_mode(size)
field_size = 25
clock = pg.time.Clock()




while True:
    dps += 1
    for event in pg.event.get():
        if event.type == pg.QUIT:
            exit(0)
        elif event.type == pg.MOUSEBUTTONDOWN:
            x_mouse, y_mouse = pg.mouse.get_pos()
            x_mouse //= nn
            y_mouse //= nn
            if event.button == 1:
                field[x_mouse][y_mouse] = 1
            if event.button == 3:
                field[x_mouse][y_mouse] = 0
            print(x_mouse, y_mouse)


    for w in range(field_size):
        for q in range(field_size):

            

            IDL = field[q][w]
            inn1[0]=q/field_size
            inn1[1]=w/field_size

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

            
            s = neu31

            if s>1:s=1
            if s<0:s=0
            if IDL!=-1:

                """ константа для формулы локального градиента нейронов (выходящий нейрон) """
                fii = error(neu31)*neu31*(1 - neu31)


                """ локалград для скрытого слоя """
                for i in range(2):
                #    print(fii, weight31[i], inn2[i])
                    localgr31.append(localgrNN(fii, weight31[i], inn2[i]))

                weight21bp = backpropagation(weight21, step, localgr31[0], inn1)
                weight22bp = backpropagation(weight22, step, localgr31[1], inn1)
                weight31bp = backpropagation(weight31, step, fii, inn2)
                


                pg.draw.rect(sc, (IDL*255, IDL*255, IDL*255), (q*nn,w*nn,nn,nn))

            else:
                if s<0.5:
                    pg.draw.rect(sc, ((1-s)*255, s, s), (q*nn,w*nn,nn,nn))
                else:
                    pg.draw.rect(sc, (s, s, s*255), (q*nn,w*nn,nn,nn))
    pg.display.update()
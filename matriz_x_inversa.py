from matriz_x import*
import numpy as np

def matriz_x_inversa(lista_lambda):


    tamanho = len(lista_lambda)

    return(matriz(lista_lambda,tamanho,tamanho))


def matriz(lista_lambda,l,c):
    mae = []
    for i in range(l):
        linha = []
        for j in range (c):

            if i == j:  
                linha.append(1)

            elif i < j:
                linha.append(0)
            else:
                linha.append(produtorio_x(lista_lambda,i,j))

        mae.append(linha)

    return(mae)


def produtorio_x(lista_lambdas,i,j):

    num = numerador(lista_lambdas, i, j)
    den = denominador(lista_lambdas, i,j)
    return item(num,den)


def numerador(lista_lambda,i,j):
    numer = 1

    for k in range(j,i):
        numer = numer * lista_lambda[k]

    sinal = (-1) ** (i-j)
    return numer * sinal

    

def denominador(lista_lambda,i,j):

    denom = 1
    for k in range(j,i):
        denom = denom * (lista_lambda[i] - lista_lambda[k])
    return denom

def item(numerador,denominador):

    if denominador == 0:
        return 0
    return numerador /denominador
    





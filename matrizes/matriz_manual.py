def matriz(l,c):
    mae = []
    for y in range(l):
        linha = []
        for x in range (c):
            linha.append(0)
        mae.append(linha)

    print(mae)

matriz(2,2)
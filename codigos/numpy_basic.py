import numpy as np
import matriz_x as mx
import matriz_x_inversa as mxx

# PASSO 1: Configurar os dados iniciais (O mundo real)
# Constantes de decaimento (lambdas fictícios)
lambda_A = 0.5
lambda_B = 0.2

# Vetor N0: Começamos com 100 átomos do Isótopo A e 0 do B
N0 = np.array([100.0, 0.0])

# Matriz Lambda: A diagonal principal é a perda (-), a de baixo é o ganho (+)
Lambda_matriz = np.array([
    [-lambda_A,  0.0],
    [ lambda_A, -lambda_B]
])

resultado_x =mx.matriz_x([0.5,0.2])




#print(Lambda_matriz)

# Tempo t que queremos calcular (ex: 5 segundos/dias)
tempo = 5.0

print("--- Iniciando Cálculo ---")

# PASSO 2: A Mágica da Diagonalização (Passo 2 do quadro)
# np.linalg.eig encontra os autovalores e a matriz X (autovetores)
autovalores, X = np.linalg.eig(Lambda_matriz)

# A matriz inversa de X (X^-1)
X_inv = np.linalg.inv(X)

# PASSO 3: O Tempo e o Decaimento (O "e" da matriz fantasma)
# np.exp calcula o e^(autovalor * tempo) para cada elemento
decaimentos = np.exp(autovalores * tempo)

# np.diag monta aquela matriz com os zeros fora da diagonal
D_tempo = np.diag(decaimentos)

# PASSO 4: A Equação Final (Caixa no canto inferior direito do quadro)
# Juntamos tudo multiplicando as matrizes com o operador '@'
# N(t) = X * D(e^-lambda*t) * X^-1 * N0
N_final = X @ D_tempo @ X_inv @ N0

#print("\nQuantidades Iniciais (A, B):")
#print(np.round(N0, 2))

#print(f"\nQuantidades após tempo t={tempo} (A, B):")
#print(np.round(N_final, 2)) 
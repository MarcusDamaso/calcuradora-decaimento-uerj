import numpy as np
import matriz_x as mx
import matriz_x_inversa as mxx
import periodictable

# Dicionário global
dicionario_Massa_molar = {
    "U-238" : periodictable.U[238].mass,
    "Th-234": periodictable.Th[234].mass,
    "Pa-234": periodictable.Pa[234].mass,
    "U-234" : periodictable.U[234].mass,
    "Th-230": periodictable.Th[230].mass,
    "Ra-226": periodictable.Ra[226].mass,
    "Rn-222": periodictable.Rn[222].mass,
    "Po-218": periodictable.Po[218].mass,
    "Pb-214": periodictable.Pb[214].mass,
    "Bi-214": periodictable.Bi[214].mass,
    "Po-214": periodictable.Po[214].mass,
    "Tl-210": periodictable.Tl[210].mass,
    "Pb-210": periodictable.Pb[210].mass,
    "Bi-210": periodictable.Bi[210].mass,
    "Po-210": periodictable.Po[210].mass,
    "Pb-206": periodictable.Pb[206].mass,
}

def calculo_N0(massa_isotopo, chave_isotopo):
    return (massa_isotopo / dicionario_Massa_molar[chave_isotopo])

# Função principal que o Streamlit vai chamar
def calcular_decaimento(elementos, massa_elementos, lista_lambdas, t):
    
    # 1. Monta as matrizes X e X^-1 (enviando a lista_lambdas)
    matriz_x_gerada = mx.matriz_x(lista_lambdas)
    matriz_xx_gerada = mxx.matriz_x_inversa(lista_lambdas)
    
    X_numpy = np.array(matriz_x_gerada, dtype=float)
    X_inv_numpy = np.array(matriz_xx_gerada, dtype=float)
    
    # 2. Monta a matriz diagonal eta1
    # Usamos o mesmo nome 'lista_lambdas' para não dar erro de "not defined"
    lambdas_array = np.array(lista_lambdas, dtype=float)
    resultados_euler = np.exp(-lambdas_array * t)
    eta1 = np.diag(resultados_euler)
    
    # 3. Monta o vetor N0 inicial
    N0_paracalculo = []
    for k in range(len(elementos)):
        chave = elementos[k]
        massa = massa_elementos[k]
        N0_paracalculo.append(calculo_N0(massa, chave))
        
    N0_numpy = np.array(N0_paracalculo)
    
    # 4. Cálculo Matricial Final
    calculo_final_mols = X_numpy @ eta1 @ X_inv_numpy @ N0_numpy
    
    # 5. Converte para Gramas
    vetor_massas_molares = np.array([dicionario_Massa_molar[chave] for chave in elementos])
    massa_final_gramas = calculo_final_mols * vetor_massas_molares
    
    return massa_final_gramas
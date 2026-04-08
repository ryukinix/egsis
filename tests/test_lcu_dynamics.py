import networkx as nx
import numpy as np
import pytest
from egsis.lcu import LabeledComponentUnfolding

def test_lcu_temporal_dynamics():
    # Cria grafo linha pequeno 0-1-2-3-4
    G = nx.path_graph(5)
    # Sementes nos extremos
    G.nodes[0]["label"] = 1
    G.nodes[4]["label"] = 2
    # Atribui pesos iniciais (necessário para a lógica)
    for u, v in G.edges:
        G.edges[u, v]["weight"] = 1.0
    
    lcu = LabeledComponentUnfolding(n_classes=2, competition_level=1.0, max_iter=10)
    
    # Executa step a step para observar a evolução
    lcu.init(G)
    
    # Verifica estado inicial
    sum_delta_initial = np.sum(lcu.delta)
    assert sum_delta_initial == 0
    
    # Executa algumas iterações
    for _ in range(5):
        lcu.step(G)
        lcu.iterations += 1
        
    sum_delta_after = np.sum(lcu.delta)
    
    # O LCU dinâmico DEVE ter acumulado partículas (delta > 0)
    assert sum_delta_after > 0, "O sistema não está evoluindo a dominância de arestas (partículas estáticas)"

if __name__ == "__main__":
    pytest.main([__file__])

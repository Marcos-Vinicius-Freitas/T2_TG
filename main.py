# Teoria dos Grafos - Trabalho 2
# Higor Vinicius e Marcos Vinicius

# Passo 1 · Importação das bibliotecas

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans

# Passo 2 · Leitura e pré-processamento do dataset

caminho = "./disney_plus_titles.csv"

dados = pd.read_csv(caminho)

print(dados.head())

def preprocessar_lista(texto):
    if pd.isna(texto):
        return []

    return [item.strip() for item in str(texto).split(',') if item.strip()]


for coluna in ['cast', 'director', 'listed_in', 'country']:
    dados[coluna] = dados[coluna].apply(preprocessar_lista)

dados['description'] = dados['description'].fillna('')


# Passo 3 · Transformação do texto em vetores TF-IDF

vetorizador = TfidfVectorizer(
    stop_words='english',
    max_features=2000,
    token_pattern=r'[A-Za-z]{2,}',
    min_df=2,
    max_df=0.8,
)

matriz_tfidf = vetorizador.fit_transform(dados['description'])

matriz_similaridade = cosine_similarity(matriz_tfidf)

print("Formato:", matriz_similaridade.shape)

matriz_temp = matriz_similaridade.copy()

np.fill_diagonal(matriz_temp, -1)

maior_similaridade = matriz_temp.max()
i, j = np.unravel_index(matriz_temp.argmax(), matriz_temp.shape)

print(f"Maior similaridade entre filmes diferentes: {maior_similaridade:.3f}")
print(f"  • {dados.iloc[i]['title']}  ×  {dados.iloc[j]['title']}")

print("  • Descrição 1:", dados.iloc[i]['description'])
print("  • Descrição 2:", dados.iloc[j]['description'])


def testar_filme_especifico(nome_filme, k=5):
    mask = dados['title'].str.contains(nome_filme, case=False, na=False, regex=False)
    if not mask.any():
        print(f"Filme '{nome_filme}' não encontrado")
        return

    indice = dados[mask].index[0]
    print(f"\n# TESTANDO: {dados.loc[indice, 'title']}")
    print(f"- Descrição: {dados.loc[indice, 'description']}")

    similaridades = matriz_similaridade[indice]

    filmes_similares = []
    for idx, sim in enumerate(similaridades):
        if idx != indice and sim > 0:
            filmes_similares.append((idx, sim, dados.loc[idx, 'title']))

    filmes_similares.sort(key=lambda t: t[1], reverse=True)

    print("\nFILMES SIMILARES (cosseno):")
    for pos, (idx, sim, tit) in enumerate(filmes_similares[:k], 1):
        print(f"   {pos}. {tit} → {sim:.3f}")


filme_escolhido = input("Filme a ser buscado: ")
testar_filme_especifico(filme_escolhido)

# Passo 4 · Agrupamento com MiniBatchKMeans

kmeans = MiniBatchKMeans(n_clusters=100, random_state=42)

clusters = kmeans.fit_predict(matriz_tfidf)

dados['cluster'] = clusters

print(dados['cluster'].value_counts().head(10))

mask_sw = dados['title'].str.contains(filme_escolhido, case=False, na=False, regex=False)
if mask_sw.any():
    numero_cluster = int(dados.loc[mask_sw, 'cluster'].iloc[0])
else:
    numero_cluster = int(dados['cluster'].value_counts().index[0])

print(f"Cluster {numero_cluster}:")
print(dados.loc[dados['cluster'] == numero_cluster, 'title'].head(15).tolist())

# Passo 5 · Construção do grafo com NetworkX

similaridade_minima = 0.15

def criar_grafo(dados, matriz_similaridade):
    print("Montando o grafo...")

    G = nx.Graph()

    for idx, filme in dados.iterrows():
        G.add_node(f"filme_{idx}", tipo="filme", titulo=filme['title'],
                   cluster=int(filme['cluster']))

    n = len(dados)
    for i in range(n):
        for j in range(i + 1, n):
            if matriz_similaridade[i, j] > similaridade_minima:
                G.add_edge(f"filme_{i}", f"filme_{j}",
                           tipo="similaridade",
                           peso=float(matriz_similaridade[i, j]))

    for idx, filme in dados.iterrows():
        for ator in filme['cast']:
            if ator:
                ator_id = f"ator_{ator}"
                G.add_node(ator_id, tipo="ator", nome=ator)
                G.add_edge(f"filme_{idx}", ator_id, tipo="elenco")

    for idx, filme in dados.iterrows():
        for diretor in filme['director']:
            if diretor:
                diretor_id = f"diretor_{diretor}"
                G.add_node(diretor_id, tipo="diretor", nome=diretor)
                G.add_edge(f"filme_{idx}", diretor_id, tipo="direcao")

    for idx, filme in dados.iterrows():
        for pais in filme['country']:
            if pais:
                pais_id = f"pais_{pais}"
                G.add_node(pais_id, tipo="pais", nome=pais)
                G.add_edge(f"filme_{idx}", pais_id, tipo="origem")

    for idx, filme in dados.iterrows():
        for categoria in filme['listed_in']:
            if categoria:
                categoria_id = f"categoria_{categoria}"
                G.add_node(categoria_id, tipo="categoria", nome=categoria)
                G.add_edge(f"filme_{idx}", categoria_id, tipo="genero")

    print("\nESTATÍSTICAS DO GRAFO")
    print("   • Vértices:", G.number_of_nodes())
    print("   • Arestas:", G.number_of_edges())
    return G

grafo = criar_grafo(dados, matriz_similaridade)

print("Vértices por tipo:", Counter(nx.get_node_attributes(grafo, 'tipo').values()))

tipos = nx.get_node_attributes(grafo, 'tipo')
atores_grau = [(no, grau) for no, grau in grafo.degree() if tipos[no] == 'ator']
atores_grau.sort(key=lambda t: t[1], reverse=True)

print("\nTop 10 atores com maior grau:")
for pos, (no, grau) in enumerate(atores_grau[:10], 1):
    print(f"   {pos}. {grafo.nodes[no]['nome']} → {grau} filmes")

print("\nO grafo é conexo?", nx.is_connected(grafo))
print("Número de componentes conexas:", nx.number_connected_components(grafo))

# Passo 6 · Desenho e visualização do subgrafo

CORES = {'filme': '#9ecae1', 'ator': '#c994c7', 'diretor': 'orange',
         'categoria': 'lightgreen', 'pais': 'yellow'}

def encontrar_no_filme(grafo, titulo):
    for no, atributos in grafo.nodes(data=True):
        if atributos.get('tipo') == 'filme' and atributos.get('titulo') == titulo:
            return no
    return None

def visualizar_grafo(grafo, titulo_filme, k=5):
    filme_node = encontrar_no_filme(grafo, titulo_filme)
    if filme_node is None:
        print(f"Filme '{titulo_filme}' não encontrado.")
        return

    filmes_similares = []
    for vizinho in grafo.neighbors(filme_node):
        if grafo.nodes[vizinho]['tipo'] == 'filme':
            peso = grafo[filme_node][vizinho]['peso']
            filmes_similares.append((vizinho, peso, grafo.nodes[vizinho]['titulo']))

    top_k = sorted(filmes_similares, key=lambda t: t[1], reverse=True)[:k]

    nos = {filme_node}
    for vizinho in grafo.neighbors(filme_node):
        if grafo.nodes[vizinho]['tipo'] != 'filme':
            nos.add(vizinho)
    for vizinho, _, _ in top_k:
        nos.add(vizinho)
        for v2 in grafo.neighbors(vizinho):
            if grafo.nodes[v2]['tipo'] != 'filme':
                nos.add(v2)

    sub = grafo.subgraph(nos)

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(sub, k=1, iterations=50, seed=42)

    cores = [CORES.get(sub.nodes[n]['tipo'], 'gray') for n in sub.nodes()]

    rotulos = {}
    for n in sub.nodes():
        if sub.nodes[n]['tipo'] == 'filme':
            titulo = sub.nodes[n]['titulo']
            rotulos[n] = titulo if len(titulo) <= 20 else titulo[:18] + "..."
        else:
            rotulos[n] = sub.nodes[n]['nome']

    nx.draw_networkx_nodes(sub, pos, node_color=cores, node_size=400, edgecolors='black')
    nx.draw_networkx_edges(sub, pos, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(sub, pos, rotulos, font_size=9)
    plt.title(f"Subgrafo de '{titulo_filme}' e seus {k} filmes mais similares", fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

titulo_teste = dados.loc[dados['title'].str.contains(filme_escolhido, case=False, na=False, regex=False)]
titulo_teste = titulo_teste['title'].iloc[0] if len(titulo_teste) else dados['title'].iloc[0]
visualizar_grafo(grafo, titulo_teste)

# Passo 7 · Recomendação com a métrica de Adamic-Adar

def recomendar_adamic_adar(grafo, titulo_filme, k=5):
    filme_node = encontrar_no_filme(grafo, titulo_filme)
    if filme_node is None:
        print(f"Filme '{titulo_filme}' não encontrado.")
        return []

    candidatos = [no for no, atributos in grafo.nodes(data=True)
                  if atributos['tipo'] == 'filme' and no != filme_node]

    pares = [(filme_node, c) for c in candidatos]

    pontuacoes = nx.adamic_adar_index(grafo, pares)

    resultado = [(grafo.nodes[v]['titulo'], pontuacao)
                 for _, v, pontuacao in pontuacoes if pontuacao > 0]
    resultado.sort(key=lambda t: t[1], reverse=True)

    return resultado[:k]


def adamic_adar_manual(grafo, u, v):
    return sum(1 / np.log(grafo.degree(z)) for z in nx.common_neighbors(grafo, u, v))

def comparar_adamic_adar(grafo, titulo_filme, k=5):
    filme_node = encontrar_no_filme(grafo, titulo_filme)
    candidatos = [no for no, a in grafo.nodes(data=True)
                  if a['tipo'] == 'filme' and no != filme_node]

    manual = sorted(((grafo.nodes[c]['titulo'], adamic_adar_manual(grafo, filme_node, c))
                     for c in candidatos), key=lambda t: t[1], reverse=True)
    manual = [(t, s) for t, s in manual if s > 0][:k]

    nativo = recomendar_adamic_adar(grafo, titulo_filme, k)

    iguais = all(abs(a[1] - b[1]) < 1e-9 for a, b in zip(manual, nativo)) and len(manual) == len(nativo)
    print("Versão manual igual à do NetworkX?", iguais)
    return manual, nativo

comparar_adamic_adar(grafo, titulo_teste)

# Passo 8 · Recomendação

def recomendar(nome_digitado, k=5):
    mask = dados['title'].str.contains(nome_digitado, case=False, na=False, regex=False)
    encontrados = dados.loc[mask, 'title'].tolist()
    if not encontrados:
        print(f"Nenhum título encontrado para '{nome_digitado}'.")
        return

    titulo = encontrados[0]
    if len(encontrados) > 1:
        print(f"{len(encontrados)} títulos encontrados. Usando: {titulo}")
        print("Outras opções:", encontrados[1:4])

    print("\n[1] Similaridade de cosseno (descrição):")
    indice = dados.index[dados['title'] == titulo][0]
    similares = [(idx, sim, dados.loc[idx, 'title'])
                 for idx, sim in enumerate(matriz_similaridade[indice])
                 if idx != indice and sim > 0]
    similares.sort(key=lambda t: t[1], reverse=True)
    for pos, (idx, sim, tit) in enumerate(similares[:k], 1):
        print(f"   {pos}. {tit} → {sim:.3f}")

    print("\n[2] Adamic-Adar (grafo):")
    for pos, (tit, pontuacao) in enumerate(recomendar_adamic_adar(grafo, titulo, k), 1):
        print(f"   {pos}. {tit} → {pontuacao:.3f}")

    visualizar_grafo(grafo, titulo)

recomendar(filme_escolhido.strip())

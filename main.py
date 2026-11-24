# %% [markdown]
# # Trabalho P2 · Teoria dos Grafos
# ## Sistema de Recomendação de Filmes/Séries com TF-IDF, K-Means e Grafos
#
# **Universidade Federal da Grande Dourados (UFGD) · FACET**
# **Bacharelado em Engenharia de Computação**
# **Autor do roteiro: Prof. Alexandre Augusto Angelo de Souza
#
# ---
#
# ### Como usar este roteiro
#
# Este notebook é um **esqueleto guiado**. Parte do código já está pronta e parte está marcada para vocês completarem:
#
# | Marcação    | Significado                                            |
# |---          |---                                                     |
# | `# TODO:`   | instrução do que deve ser feito naquela linha ou bloco |
# | `...`       | lugar onde vocês devem escrever o código |
# | `# DICA:`   | pista sobre a função ou o raciocínio esperado |
# | `# PERGUNTA:` | questão para responder na documentação do trabalho |
#
# Executem as células **na ordem**. Ao final de cada passo existe uma pequena verificação para conferir se está tudo certo antes de seguir.
#
# **Entrega (Google Classroom):** código Python, prints da execução e dos resultados, e a documentação respondendo às perguntas marcadas.

# %% [markdown]
# ## Passo 1 · Importação das bibliotecas
#
# Precisamos de bibliotecas para: manipular dados (tabelas), criar e analisar grafos, desenhar gráficos, fazer contas com vetores e aplicar aprendizado de máquina sobre texto.

# %%
# Se estiver no Google Colab, monte o Drive para acessar o arquivo CSV
# from google.colab import drive
# drive.mount('/content/drive')

# %%
import pandas as pd                 # manipulação de dados (DataFrame)
import numpy as np                  # cálculos numéricos
import networkx as nx               # criação e análise de grafos
import matplotlib.pyplot as plt     # visualização

# TODO: importe do scikit-learn as três ferramentas abaixo
# DICA: estão nos módulos sklearn.feature_extraction.text, sklearn.metrics.pairwise e sklearn.cluster
from sklearn.feature_extraction.text import TfidfVectorizer   # vetorizador TF-IDF
from sklearn.metrics.pairwise import cosine_similarity          # similaridade por cosseno
from sklearn.cluster import MiniBatchKMeans                   # versão "em lotes" do K-Means

# %% [markdown]
# ## Passo 2 · Leitura e pré-processamento do dataset
#
# O arquivo `disney_plus_titles.csv` possui colunas como `title`, `director`, `cast`, `country`, `listed_in` e `description`.
#
# Problema: as colunas `cast`, `director`, `country` e `listed_in` guardam **vários valores em um único texto**, separados por vírgula (ex.: `"Mark Hamill, Harrison Ford, Carrie Fisher"`). Para o grafo, precisamos que cada ator, diretor, país e categoria seja um **item separado** de uma lista.

# %%
# TODO: ajuste o caminho para onde VOCÊ salvou o arquivo no Drive
caminho = "disney_plus_titles.csv"

# TODO: leia o CSV com o pandas
dados = ...

display(dados.head())

# %%
# --- Análise exploratória rápida ---
# TODO: mostre os nomes das colunas
print("Colunas:", ...)

# TODO: mostre quantas linhas e colunas existem
print("Formato:", ...)

# TODO: conte quantas descrições estão faltando (valores nulos)
# DICA: .isna().sum()
print("Descrições faltando:", ...)

# Exemplo de uma descrição
print("Exemplo:", dados['description'].iloc[0])

# %%
def preprocessar_lista(texto):
    """Converte 'A, B, C' em ['A', 'B', 'C']. Valores nulos viram lista vazia."""
    # TODO: se o valor for nulo, retorne uma lista vazia
    # DICA: pd.isna(texto)
    if ...:
        return ...

    # TODO: separe o texto pela vírgula e remova os espaços extras de cada item
    # DICA: str(texto).split(',') e item.strip() dentro de uma list comprehension
    return [...]


# Teste rápido da função (deve imprimir ['Mark Hamill', 'Harrison Ford'] e [])
print(preprocessar_lista("Mark Hamill,  Harrison Ford"))
print(preprocessar_lista(np.nan))

# %%
# TODO: aplique a função nas colunas com múltiplos valores
# DICA: dados['coluna'] = dados['coluna'].apply(preprocessar_lista)
for coluna in ['cast', 'director', 'listed_in', 'country']:
    ...

# TODO: preencha descrições vazias com string vazia (o TF-IDF não aceita NaN)
dados['description'] = ...

# Verificação
dados[['title', 'cast', 'director', 'listed_in', 'country']].head(3)

# %% [markdown]
# ## Passo 3 · Transformação do texto em vetores TF-IDF
#
# Algoritmos de aprendizado de máquina trabalham com números, não com texto. O **TF-IDF** transforma cada descrição em um vetor onde cada posição corresponde a uma palavra do vocabulário e o valor indica o quanto aquela palavra é importante **naquele filme** em relação ao **catálogo inteiro**.
#
# $$\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log\frac{N}{\text{DF}(t)}$$
#
# Lembrem: as descrições estão **em inglês**, então as *stop words* devem ser do inglês.

# %%
# TODO: crie o vetorizador TF-IDF configurando os parâmetros indicados
vetorizador = TfidfVectorizer(
    stop_words= 'english',             # remover palavras muito comuns do idioma das descrições
    max_features= 2000,           # limitar o vocabulário (sugestão: 2000 palavras)
    token_pattern=r'[A-Za-z]{2,}',  # só palavras com letras e pelo menos 2 caracteres
    min_df= 2,                 # ignorar palavras que aparecem em menos de 2 filmes
    max_df= 0.8,                 # ignorar palavras presentes em mais de 80% dos filmes
)

# TODO: ajuste o vetorizador e transforme a coluna 'description'
# DICA: método fit_transform
matriz_tfidf = vetorizador.fit_transform(dados['description'])

print("Formato da matriz TF-IDF:", matriz_tfidf.shape)
print("   • Número de filmes:", matriz_tfidf.shape[0])
print("   • Tamanho do vocabulário:", matriz_tfidf.shape[1])

# TODO: mostre as 20 primeiras palavras do vocabulário
# DICA: vetorizador.get_feature_names_out()
print(vetorizador.get_feature_names_out()[:20])

# PERGUNTA: por que a matriz é "esparsa"? O que significam a maioria dos valores serem zero?

# %% [markdown]
# ### 3.1 · Similaridade por cosseno
#
# Com cada filme representado por um vetor, medimos o quanto dois filmes se parecem pelo **cosseno do ângulo** entre seus vetores:
#
# $$\text{sim}(A,B) = \frac{A \cdot B}{\|A\|\,\|B\|}$$
#
# Valor 1 significa descrições "idênticas" em termos de palavras, valor 0 significa nenhuma palavra relevante em comum.

# %%
# TODO: calcule a matriz de similaridade entre TODOS os pares de filmes
matriz_similaridade = ...

print("Formato:", matriz_similaridade.shape)   # deve ser (n_filmes, n_filmes)

# --- Encontrar o par de filmes DIFERENTES mais parecido ---
matriz_temp = matriz_similaridade.copy()

# TODO: a diagonal principal vale 1 (cada filme com ele mesmo). Substitua-a por -1
# DICA: np.fill_diagonal
...

# TODO: encontre o maior valor e a posição (i, j) onde ele ocorre
# DICA: .max(), .argmax() e np.unravel_index
maior_similaridade = ...
i, j = ...

print(f"Maior similaridade entre filmes diferentes: {maior_similaridade:.3f}")
print(f"  • {dados.iloc[i]['title']}  ×  {dados.iloc[j]['title']}")

# PERGUNTA: se o valor encontrado for 1.0, o que isso indica sobre as descrições desses títulos?
# Confiram no dataset e comentem na documentação.

# %%
def testar_filme_especifico(nome_filme, k=5):
    """Mostra os k filmes com maior similaridade de cosseno (somente texto, sem grafo)."""
    # TODO: localize o filme pelo título (busca parcial e sem diferenciar maiúsculas)
    # DICA: dados['title'].str.contains(nome_filme, case=False, na=False)
    mask = ...
    if not mask.any():
        print(f"Filme '{nome_filme}' não encontrado")
        return

    indice = dados[mask].index[0]
    print(f"\n# TESTANDO: {dados.loc[indice, 'title']}")
    print(f"- Descrição: {dados.loc[indice, 'description']}")

    # TODO: pegue a linha da matriz de similaridade correspondente ao filme
    similaridades = ...

    # TODO: monte uma lista de tuplas (indice, similaridade, titulo)
    #       excluindo o próprio filme e similaridades iguais a zero
    filmes_similares = []
    ...

    # TODO: ordene da maior para a menor similaridade
    ...

    print("\nFILMES SIMILARES (cosseno):")
    for pos, (idx, sim, tit) in enumerate(filmes_similares[:k], 1):
        print(f"   {pos}. {tit} → {sim:.3f}")


testar_filme_especifico("Star Wars")

# %% [markdown]
# ## Passo 4 · Agrupamento com MiniBatchKMeans
#
# O K-Means separa os filmes em *k* grupos (clusters) de descrições parecidas. O **MiniBatchKMeans** faz o mesmo processando pequenos lotes por vez, sendo bem mais rápido em bases grandes. Sugestão do enunciado: **100 clusters**.

# %%
# TODO: crie o modelo com 100 clusters e random_state=42 (para resultados reproduzíveis)
kmeans = ...

# TODO: treine o modelo e obtenha o cluster de cada filme em um único comando
# DICA: fit_predict sobre a matriz TF-IDF
clusters = ...

# TODO: salve o resultado em uma nova coluna 'cluster' do DataFrame
...

# Verificação: quantos filmes há em cada cluster (10 maiores)
print(dados['cluster'].value_counts().head(10))

# %%
# Explorando um cluster: quais títulos ficaram juntos?
# TODO: escolha o número de um cluster e mostre os títulos que pertencem a ele
numero_cluster = ...
print(dados.loc[dados['cluster'] == numero_cluster, 'title'].head(15).tolist())

# PERGUNTA: os títulos desse cluster fazem sentido juntos? Que tema eles compartilham?

# %% [markdown]
# ## Passo 5 · Construção do grafo com NetworkX
#
# Vamos montar um **grafo não direcionado e heterogêneo**, isto é, com vértices de tipos diferentes:
#
# | Tipo de vértice | Identificador sugerido | Atributos |
# |---|---|---|
# | filme | `filme_<indice>` | `tipo`, `titulo`, `cluster` |
# | ator | `ator_<nome>` | `tipo`, `nome` |
# | diretor | `diretor_<nome>` | `tipo`, `nome` |
# | país | `pais_<nome>` | `tipo`, `nome` |
# | categoria | `categoria_<nome>` | `tipo`, `nome` |
#
# | Aresta | Entre | Atributo |
# |---|---|---|
# | similaridade | filme ↔ filme (se cosseno > limiar) | `peso` = similaridade |
# | elenco | filme ↔ ator | `tipo="elenco"` |
# | direção | filme ↔ diretor | `tipo="direcao"` |
# | origem | filme ↔ país | `tipo="origem"` |
# | gênero | filme ↔ categoria | `tipo="genero"` |
#
# **Importante:** o mesmo ator que aparece em dois filmes deve ser **um único vértice**. É isso que cria os caminhos entre filmes no grafo.

# %%
similaridade_minima = 0.15   # limiar para ligar dois filmes por similaridade de texto

def criar_grafo(dados, matriz_similaridade):
    print("Montando o grafo...")

    # TODO: crie um grafo NÃO direcionado vazio
    G = ...

    # ---- 5.1 Vértices de filmes ----
    for idx, filme in dados.iterrows():
        # TODO: adicione o vértice f"filme_{idx}" com os atributos tipo, titulo e cluster
        G.add_node(...)

    # ---- 5.2 Arestas filme ↔ filme por similaridade ----
    n = len(dados)
    for i in range(n):
        # TODO: percorra apenas j > i para não repetir pares nem ligar o filme a ele mesmo
        for j in range(...):
            # TODO: se a similaridade entre i e j for maior que o limiar, crie a aresta
            #       guardando tipo="similaridade" e peso=<valor da similaridade>
            ...

    # ---- 5.3 Atores ----
    for idx, filme in dados.iterrows():
        for ator in filme['cast']:
            if ator:
                ator_id = f"ator_{ator}"
                # TODO: adicione o vértice do ator (tipo="ator", nome=ator)
                ...
                # TODO: ligue o filme ao ator (tipo="elenco")
                ...

    # ---- 5.4 Diretores ----
    # TODO: repita a lógica dos atores usando a coluna 'director'
    #       prefixo "diretor_", tipo="diretor", aresta tipo="direcao"
    ...

    # ---- 5.5 Países ----
    # TODO: repita usando a coluna 'country' (prefixo "pais_", aresta tipo="origem")
    ...

    # ---- 5.6 Categorias ----
    # TODO: repita usando a coluna 'listed_in' (prefixo "categoria_", aresta tipo="genero")
    ...

    print("\nESTATÍSTICAS DO GRAFO")
    # TODO: imprima o total de vértices e de arestas
    # DICA: G.number_of_nodes() e G.number_of_edges()
    print("   • Vértices:", ...)
    print("   • Arestas:", ...)
    return G


grafo = criar_grafo(dados, matriz_similaridade)

# %%
# --- Análise estrutural do grafo (use nos resultados da documentação) ---
from collections import Counter

# TODO: conte quantos vértices existem de cada tipo
# DICA: Counter(nx.get_node_attributes(grafo, 'tipo').values())
print("Vértices por tipo:", ...)

# TODO: liste os 10 atores com MAIOR GRAU (mais filmes no catálogo)
# DICA: percorra grafo.degree() filtrando os vértices de tipo 'ator'
...

# PERGUNTA: o grafo é conexo? Quantas componentes conexas ele possui?
# DICA: nx.is_connected / nx.number_connected_components

# %% [markdown]
# ## Passo 6 · Desenho e visualização do subgrafo
#
# O grafo completo tem milhares de vértices e é impossível de visualizar. Vamos desenhar apenas um **subgrafo** com:
#
# 1. o filme escolhido e seus vizinhos (atores, diretores, países, categorias);
# 2. os 5 filmes mais similares a ele (arestas de similaridade) e os vizinhos desses filmes.

# %%
CORES = {'filme': '#9ecae1', 'ator': '#c994c7', 'diretor': 'orange',
         'categoria': 'lightgreen', 'pais': 'yellow'}

def encontrar_no_filme(grafo, titulo):
    """Retorna o identificador do vértice cujo atributo 'titulo' é igual ao informado."""
    # TODO: percorra grafo.nodes(data=True) e devolva o nó do tipo 'filme' com esse título
    ...
    return None


def visualizar_grafo(grafo, titulo_filme, k=5):
    filme_node = encontrar_no_filme(grafo, titulo_filme)
    if filme_node is None:
        print(f"Filme '{titulo_filme}' não encontrado.")
        return

    # TODO: entre os vizinhos do filme, selecione os que são do tipo 'filme'
    #       e guarde (vizinho, peso, titulo). O peso está em grafo[filme_node][vizinho]['peso']
    filmes_similares = []
    ...

    # TODO: ordene por peso decrescente e pegue os k primeiros
    top_k = ...

    # TODO: monte a lista de vértices do subgrafo:
    #       o filme principal + seus vizinhos não-filme
    #       + cada filme do top_k + os vizinhos não-filme de cada um deles
    nos = {filme_node}
    ...

    # TODO: crie o subgrafo induzido por esses vértices
    sub = ...

    # ---- desenho ----
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(sub, k=1, iterations=50, seed=42)

    cores = [CORES.get(sub.nodes[n]['tipo'], 'gray') for n in sub.nodes()]

    # TODO: monte o dicionário de rótulos: título (abreviado) para filmes e nome para os demais
    rotulos = {}
    ...

    nx.draw_networkx_nodes(sub, pos, node_color=cores, node_size=400, edgecolors='black')
    nx.draw_networkx_edges(sub, pos, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(sub, pos, rotulos, font_size=9)
    plt.title(f"Subgrafo de '{titulo_filme}' e seus {k} filmes mais similares", fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# TODO: teste com um título exato do catálogo
visualizar_grafo(grafo, "...")

# %% [markdown]
# ## Passo 7 · Recomendação com a métrica de Adamic-Adar
#
# Até aqui, a similaridade usou **apenas o texto**. Agora vamos usar a **estrutura do grafo**: dois filmes são parecidos se compartilham muitos vizinhos (atores, diretores, categorias, países, filmes similares).
#
# A métrica de **Adamic-Adar** soma os vizinhos em comum, mas dá **mais peso aos vizinhos raros**:
#
# $$AA(x,y) = \sum_{z \,\in\, N(x) \cap N(y)} \frac{1}{\log |N(z)|}$$
#
# **Intuição:** compartilhar a categoria *"Family"* (ligada a centenas de filmes) diz pouco. Compartilhar um diretor que fez só 3 filmes diz muito. O $\log$ do grau penaliza vizinhos muito populares.
#
# O NetworkX já implementa a métrica: `nx.adamic_adar_index(G, pares)` recebe uma lista de pares `(u, v)` e devolve tuplas `(u, v, pontuacao)`.
#
# **Atenção:** um vizinho comum com grau 1 gera $\log 1 = 0$ (divisão por zero). Isso não ocorre aqui, pois um vizinho comum a dois filmes tem grau ≥ 2.

# %%
def recomendar_adamic_adar(grafo, titulo_filme, k=5):
    """Retorna os k filmes com maior índice de Adamic-Adar em relação ao filme informado."""
    filme_node = encontrar_no_filme(grafo, titulo_filme)
    if filme_node is None:
        print(f"Filme '{titulo_filme}' não encontrado.")
        return []

    # TODO: crie a lista de candidatos, isto é, todos os vértices do tipo 'filme'
    #       diferentes do filme de entrada
    candidatos = ...

    # TODO: monte os pares (filme_node, candidato)
    pares = ...

    # TODO: calcule o índice de Adamic-Adar para todos os pares
    # DICA: nx.adamic_adar_index(grafo, pares)
    pontuacoes = ...

    # TODO: transforme o resultado em uma lista (titulo, pontuacao),
    #       descarte pontuação zero e ordene em ordem decrescente
    resultado = ...

    return resultado[:k]


# DESAFIO (opcional, vale como diferencial):
# Implementem a métrica "na mão" usando nx.common_neighbors e grafo.degree,
# e comparem com o resultado do NetworkX para verificar se são iguais.

# %% [markdown]
# ## Passo 8 · Sistema de recomendação final
#
# Ao receber o nome de um filme ou série, o sistema deve indicar os **5 títulos com maiores índices de similaridade**. Apresentem lado a lado os dois critérios estudados para poder compará-los na documentação.

# %%
def recomendar(nome_digitado, k=5):
    # TODO: encontre títulos que CONTENHAM o texto digitado (sem diferenciar maiúsculas)
    encontrados = ...
    if not encontrados:
        print(f"Nenhum título encontrado para '{nome_digitado}'.")
        return

    titulo = encontrados[0]
    if len(encontrados) > 1:
        print(f"{len(encontrados)} títulos encontrados. Usando: {titulo}")
        print("Outras opções:", encontrados[1:4])

    # --- Critério 1: similaridade de cosseno (texto) ---
    print("\n[1] Similaridade de cosseno (descrição):")
    # TODO: reaproveite a lógica do Passo 3 para listar os k mais similares
    ...

    # --- Critério 2: Adamic-Adar (estrutura do grafo) ---
    print("\n[2] Adamic-Adar (grafo):")
    # TODO: chame recomendar_adamic_adar e imprima o ranking com a pontuação
    ...

    # --- Visualização ---
    # TODO: chame visualizar_grafo para o título escolhido
    ...


nome = input("Digite o nome do filme/série: ").strip()
recomendar(nome)

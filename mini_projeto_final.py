# ============================================================
# MINI-PROJETO AVALIATIVO - MNIST
# Classificacao de Digitos Manuscritos
# ============================================================

#comando para instalar as bibliotecas:
#python -m pip install -r requirements.txt

#comando para rodar o projeto .py
#python mini_projeto_final.py

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image, ImageOps

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURACOES
# ============================================================

RANDOM_STATE = 42
PASTA_IMAGENS = Path("data/handwritten")
CLASSES_OCULTADAS = [4, 7]


# ============================================================
# FUNCOES AUXILIARES
# ============================================================

def calcular_metricas(y_true, y_pred):
    """Calcula as metricas exigidas no projeto."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_Weighted": precision_score(
            y_true, y_pred, average="weighted"
        ),
        "Recall_Weighted": recall_score(
            y_true, y_pred, average="weighted"
        ),
        "F1_Weighted": f1_score(
            y_true, y_pred, average="weighted"
        ),
    }


def maiores_confusoes(cm, quantidade=3):
    """Retorna as principais confusoes, ignorando a diagonal."""
    matriz = cm.copy()

    for i in range(matriz.shape[0]):
        matriz[i, i] = 0

    indices = np.argsort(matriz.ravel())[::-1]
    resultados = []

    for indice in indices:
        linha, coluna = np.unravel_index(indice, matriz.shape)
        erros = int(matriz[linha, coluna])

        if erros > 0:
            resultados.append(
                f"{int(linha)} -> {int(coluna)}: {erros} erros"
            )

        if len(resultados) == quantidade:
            break

    return resultados


def taxa_erro_por_classe(cm):
    """Calcula a taxa de erro de cada classe."""
    taxas = []

    for classe in range(cm.shape[0]):
        total = cm[classe].sum()
        corretos = cm[classe, classe]
        taxas.append((total - corretos) / total)

    return taxas


def plotar_matriz_confusao(cm, titulo):
    """Exibe uma matriz de confusao 10 x 10."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
    )
    plt.xlabel("Predicao")
    plt.ylabel("Valor verdadeiro")
    plt.title(titulo)
    plt.tight_layout()
    plt.show()


def preprocessar_imagem(caminho):
    """
    Converte uma imagem manuscrita para o formato de entrada do MNIST.

    Etapas: escala de cinza, inversao de cores, bounding box,
    redimensionamento preservando proporcao, centralizacao da massa,
    normalizacao [0, 1] e transformacao para 1 x 784.
    """
    imagem = Image.open(caminho).convert("L")
    imagem_invertida = ImageOps.invert(imagem)
    array_imagem = np.array(imagem_invertida)

    # Limiar para separar o traco do fundo.
    pixels = np.argwhere(array_imagem > 30)

    if len(pixels) == 0:
        raise ValueError("Nenhum digito foi identificado na imagem.")

    y_min, x_min = pixels.min(axis=0)
    y_max, x_max = pixels.max(axis=0)

    imagem_recortada = imagem_invertida.crop(
        (x_min, y_min, x_max + 1, y_max + 1)
    )

    # Redimensiona mantendo a proporcao e deixando margem na matriz 28 x 28.
    largura, altura = imagem_recortada.size

    if largura > altura:
        nova_largura = 20
        nova_altura = max(1, int(altura * 20 / largura))
    else:
        nova_altura = 20
        nova_largura = max(1, int(largura * 20 / altura))

    imagem_redimensionada = imagem_recortada.resize(
        (nova_largura, nova_altura)
    )

    imagem_28 = Image.new("L", (28, 28), 0)
    pos_x = (28 - nova_largura) // 2
    pos_y = (28 - nova_altura) // 2
    imagem_28.paste(imagem_redimensionada, (pos_x, pos_y))

    # Centralizacao pela massa dos pixels.
    array_28 = np.array(imagem_28, dtype=float)
    massa_total = array_28.sum()

    if massa_total == 0:
        raise ValueError("A imagem ficou sem pixels apos o processamento.")

    y_indices, x_indices = np.indices(array_28.shape)
    centro_x = (x_indices * array_28).sum() / massa_total
    centro_y = (y_indices * array_28).sum() / massa_total

    deslocamento_x = int(round(13.5 - centro_x))
    deslocamento_y = int(round(13.5 - centro_y))

    imagem_centralizada = Image.new("L", (28, 28), 0)
    imagem_centralizada.paste(
        imagem_28,
        (deslocamento_x, deslocamento_y),
    )

    # Normalizacao para [0, 1].
    imagem_normalizada = np.array(imagem_centralizada) / 255.0
    imagem_modelo = imagem_normalizada.reshape(1, 784)

    return imagem_centralizada, imagem_modelo


def obter_digito_do_nome(caminho):
    """Extrai o primeiro digito do nome, ex.: digito_5.png -> 5."""
    correspondencia = re.search(r"(\d)", caminho.stem)
    return int(correspondencia.group(1)) if correspondencia else None


def executar_fase_5_3(modelo_svm_prob, pasta):
    """Processa e classifica todas as imagens manuscritas da pasta."""
    if not pasta.exists():
        print(f"\nPasta nao encontrada: {pasta}")
        print("Crie a pasta e coloque suas imagens nela.")
        return pd.DataFrame()

    arquivos = []
    for extensao in ("*.png", "*.jpg", "*.jpeg"):
        arquivos.extend(pasta.glob(extensao))
    arquivos = sorted(arquivos)

    print(f"\nImagens manuscritas encontradas: {len(arquivos)}")

    resultados = []

    for caminho in arquivos:
        try:
            imagem_processada, imagem_modelo = preprocessar_imagem(caminho)
            predicao = modelo_svm_prob.predict(imagem_modelo)[0]
            probabilidades = modelo_svm_prob.predict_proba(imagem_modelo)[0]
            digito_real = obter_digito_do_nome(caminho)
            confianca = probabilidades[predicao]
            acertou = (
                predicao == digito_real
                if digito_real is not None
                else None
            )

            resultados.append(
                {
                    "Imagem": caminho.name,
                    "Digito_Real": digito_real,
                    "Predicao": predicao,
                    "Confianca": confianca,
                    "Acertou": acertou,
                }
            )

            # Imagem processada + probabilidades.
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].imshow(imagem_processada, cmap="gray")
            axes[0].axis("off")
            axes[0].set_title(
                f"{caminho.name}\n"
                f"Real: {digito_real} | Predicao: {predicao}"
                if digito_real is not None
                else f"{caminho.name}\nPredicao: {predicao}"
            )

            axes[1].bar(range(10), probabilidades)
            axes[1].set_xticks(range(10))
            axes[1].set_xlabel("Digito")
            axes[1].set_ylabel("Probabilidade")
            axes[1].set_title(
                f"Probabilidades de saida\n"
                f"Confianca: {confianca:.2%}"
            )

            plt.tight_layout()
            plt.show()

        except Exception as erro:
            print(f"Erro ao processar {caminho.name}: {erro}")

    resultados_df = pd.DataFrame(resultados)

    if not resultados_df.empty:
        resultados_df["Confianca"] = (
            resultados_df["Confianca"] * 100
        ).round(2)

        print("\nRESULTADOS DAS IMAGENS MANUSCRITAS")
        print(resultados_df.to_string(index=False))

        com_rotulo = resultados_df[
            resultados_df["Acertou"].notna()
        ]

        if not com_rotulo.empty:
            acuracia_propria = com_rotulo["Acertou"].mean()
            print(
                f"\nAcuracia nas imagens proprias: "
                f"{acuracia_propria:.2%}"
            )

    return resultados_df


def main():
    """Executa o projeto completo, da Fase 1 a Fase 5.3."""

    # ========================================================
    # FASE 1 - MNIST E EDA
    # ========================================================
    print("\n" + "=" * 70)
    print("FASE 1 - CARREGAMENTO E ANALISE EXPLORATORIA")
    print("=" * 70)

    print("\nBaixando/carregando o dataset MNIST...")
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="auto",
    )

    X, y = mnist.data, mnist.target
    y = y.astype(np.uint8)

    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(f"Classes: {np.unique(y)}")
    print("\nQuantidade de imagens por classe:")
    print(pd.Series(y).value_counts().sort_index())

    # Exemplo individual.
    digito_exemplo = X[9].reshape(28, 28)
    plt.figure(figsize=(3, 3))
    plt.imshow(digito_exemplo, cmap="binary")
    plt.title(f"Rotulo real: {y[9]}")
    plt.axis("off")
    plt.show()

    # Grade 2 x 5.
    fig, axes = plt.subplots(2, 5, figsize=(8, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i].reshape(28, 28), cmap="binary")
        ax.set_title(f"Digito: {y[i]}")
        ax.axis("off")
    plt.suptitle("Exemplos de imagens do MNIST", fontsize=14)
    plt.tight_layout()
    plt.show()

    print("""
INTERPRETACAO DOS DADOS

O MNIST possui 70.000 imagens de digitos manuscritos em tons de cinza.
Cada imagem possui 28 x 28 pixels, totalizando 784 caracteristicas.
Os pixels possuem valores entre 0 e 255. Na representacao do dataset,
0 corresponde ao preto e 255 ao branco, com valores intermediarios
representando diferentes tons de cinza.

Os modelos utilizados recebem vetores de caracteristicas. Por isso,
cada imagem 28 x 28 e representada como um vetor de 784 valores.
""")

    # ========================================================
    # FASE 2 - SPLIT E NORMALIZACAO
    # ========================================================
    print("\n" + "=" * 70)
    print("FASE 2 - PRE-PROCESSAMENTO E DIVISAO DOS DADOS")
    print("=" * 70)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.125,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    print(f"Treino: {X_train.shape}")
    print(f"Validacao: {X_val.shape}")
    print(f"Teste: {X_test.shape}")

    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    print(f"\nMenor valor apos normalizacao: {X_train.min()}")
    print(f"Maior valor apos normalizacao: {X_train.max()}")

    print("""
JUSTIFICATIVA DA NORMALIZACAO

Os pixels foram transformados do intervalo [0, 255] para [0, 1].
A normalizacao reduz a escala numerica das caracteristicas e contribui
para um treinamento mais estavel. Ela e especialmente relevante para
o SVM com kernel RBF, sensivel a relacoes de distancia, e para a MLP,
cujo treinamento utiliza otimizacao baseada em gradiente.
""")

    # ========================================================
    # FASE 3 - TRES MODELOS
    # ========================================================
    print("\n" + "=" * 70)
    print("FASE 3 - TREINAMENTO DOS TRES MODELOS")
    print("=" * 70)

    # SVM.
    print("\n--- SVM ---")
    modelo_svm = SVC(C=10, kernel="rbf")
    inicio = time.time()
    modelo_svm.fit(X_train, y_train)
    tempo_svm = time.time() - inicio
    acuracia_svm_val = modelo_svm.score(X_val, y_val)
    print(f"Tempo: {tempo_svm:.2f} segundos")
    print(f"Acuracia na validacao: {acuracia_svm_val:.4f}")

    # Random Forest.
    print("\n--- RANDOM FOREST ---")
    modelo_rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    inicio = time.time()
    modelo_rf.fit(X_train, y_train)
    tempo_rf = time.time() - inicio
    acuracia_rf_val = modelo_rf.score(X_val, y_val)
    print(f"Tempo: {tempo_rf:.2f} segundos")
    print(f"Acuracia na validacao: {acuracia_rf_val:.4f}")

    # MLP.
    print("\n--- MLP ---")
    modelo_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        learning_rate_init=0.001,
        max_iter=20,
        random_state=RANDOM_STATE,
    )
    inicio = time.time()
    modelo_mlp.fit(X_train, y_train)
    tempo_mlp = time.time() - inicio
    acuracia_mlp_val = modelo_mlp.score(X_val, y_val)
    print(f"Tempo: {tempo_mlp:.2f} segundos")
    print(f"Acuracia na validacao: {acuracia_mlp_val:.4f}")

    resultados_validacao = pd.DataFrame({
        "Modelo": ["SVM", "Random Forest", "MLP"],
        "Accuracy_Validacao": [
            acuracia_svm_val,
            acuracia_rf_val,
            acuracia_mlp_val,
        ],
        "Tempo_Treinamento_s": [
            tempo_svm,
            tempo_rf,
            tempo_mlp,
        ],
    })

    print("\nRESULTADOS DA VALIDACAO")
    print(resultados_validacao.to_string(index=False))

    # ========================================================
    # FASE 4 - TESTE E MATRIZES
    # ========================================================
    print("\n" + "=" * 70)
    print("FASE 4 - AVALIACAO NO CONJUNTO DE TESTE")
    print("=" * 70)

    y_pred_svm = modelo_svm.predict(X_test)
    y_pred_rf = modelo_rf.predict(X_test)
    y_pred_mlp = modelo_mlp.predict(X_test)

    metricas_svm = calcular_metricas(y_test, y_pred_svm)
    metricas_rf = calcular_metricas(y_test, y_pred_rf)
    metricas_mlp = calcular_metricas(y_test, y_pred_mlp)

    resultados = pd.DataFrame(
        [metricas_svm, metricas_rf, metricas_mlp],
        index=["SVM", "Random Forest", "MLP"],
    )

    resultados["Tempo_Treinamento_s"] = [
        tempo_svm,
        tempo_rf,
        tempo_mlp,
    ]

    print("\nMETRICAS NO TESTE")
    tabela = resultados.copy()
    colunas_metricas = [
        "Accuracy",
        "Precision_Weighted",
        "Recall_Weighted",
        "F1_Weighted",
    ]
    tabela[colunas_metricas] = (
        tabela[colunas_metricas] * 100
    ).round(2)
    tabela["Tempo_Treinamento_s"] = (
        tabela["Tempo_Treinamento_s"].round(2)
    )
    print(tabela.to_string())

    print("\nCLASSIFICATION REPORT - SVM")
    print(classification_report(y_test, y_pred_svm))

    print("\nCLASSIFICATION REPORT - RANDOM FOREST")
    print(classification_report(y_test, y_pred_rf))

    print("\nCLASSIFICATION REPORT - MLP")
    print(classification_report(y_test, y_pred_mlp))

    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)

    plotar_matriz_confusao(cm_svm, "Matriz de Confusao - SVM")
    plotar_matriz_confusao(cm_rf, "Matriz de Confusao - Random Forest")
    plotar_matriz_confusao(cm_mlp, "Matriz de Confusao - MLP")

    print("\nMAIORES CONFUSOES")
    print("SVM:", maiores_confusoes(cm_svm))
    print("Random Forest:", maiores_confusoes(cm_rf))
    print("MLP:", maiores_confusoes(cm_mlp))

    taxas_svm = taxa_erro_por_classe(cm_svm)
    taxas_rf = taxa_erro_por_classe(cm_rf)
    taxas_mlp = taxa_erro_por_classe(cm_mlp)

    classe_maior_erro_svm = np.argmax(taxas_svm)
    classe_maior_erro_rf = np.argmax(taxas_rf)
    classe_maior_erro_mlp = np.argmax(taxas_mlp)

    print("\nMAIOR TAXA DE ERRO POR CLASSE")
    print(
        f"SVM: digito {classe_maior_erro_svm} "
        f"({taxas_svm[classe_maior_erro_svm]:.2%})"
    )
    print(
        f"Random Forest: digito {classe_maior_erro_rf} "
        f"({taxas_rf[classe_maior_erro_rf]:.2%})"
    )
    print(
        f"MLP: digito {classe_maior_erro_mlp} "
        f"({taxas_mlp[classe_maior_erro_mlp]:.2%})"
    )

    melhor_modelo = resultados["F1_Weighted"].idxmax()
    print(f"\nMelhor modelo pelo F1-Score ponderado: {melhor_modelo}")

    print("""
CONCLUSAO DA FASE 4

A avaliacao no conjunto de teste permite comparar os tres modelos
em dados independentes do treinamento. A escolha do melhor modelo
considera Accuracy, Precision, Recall, F1-Score, matriz de confusao
e custo computacional.
""")

    # ========================================================
    # FASE 5.1 - CLASS MASKING
    # ========================================================
    print("\n" + "=" * 70)
    print("FASE 5.1 - CLASS MASKING")
    print("=" * 70)

    mascara_treino = ~np.isin(y_train, CLASSES_OCULTADAS)
    X_train_masked = X_train[mascara_treino]
    y_train_masked = y_train[mascara_treino]

    print(f"Classes ocultadas: {CLASSES_OCULTADAS}")
    print(f"Treino original: {X_train.shape}")
    print(f"Treino mascarado: {X_train_masked.shape}")
    print("Classes presentes:", np.unique(y_train_masked))

    modelo_svm_masked = SVC(C=10, kernel="rbf")
    inicio = time.time()
    modelo_svm_masked.fit(X_train_masked, y_train_masked)
    tempo_svm_masked = time.time() - inicio

    print(f"Tempo SVM mascarado: {tempo_svm_masked:.2f} segundos")
    print("Classes aprendidas:", modelo_svm_masked.classes_)

    # ========================================================
    # FASE 5.2 - OOD
    # ========================================================
    print("\n" + "=" * 70)
    print("FASE 5.2 - TESTE OOD")
    print("=" * 70)

    mascara_ood = np.isin(y_test, CLASSES_OCULTADAS)
    X_ood = X_test[mascara_ood]
    y_ood = y_test[mascara_ood]
    y_pred_ood = modelo_svm_masked.predict(X_ood)

    print(f"Dimensao do conjunto OOD: {X_ood.shape}")
    print("Classes verdadeiras:", np.unique(y_ood))
    print("\nQuantidade por classe:")
    print(pd.Series(y_ood).value_counts().sort_index())

    tabela_ood = pd.crosstab(y_ood, y_pred_ood).reindex(
        index=CLASSES_OCULTADAS,
        columns=[0, 1, 2, 3, 5, 6, 8, 9],
        fill_value=0,
    )

    print("\nTabela OOD:")
    print(tabela_ood)

    plt.figure(figsize=(10, 4))
    sns.heatmap(tabela_ood, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Classe prevista pelo modelo")
    plt.ylabel("Classe verdadeira")
    plt.title("Inferencia OOD - Classes ocultadas (4 e 7)")
    plt.tight_layout()
    plt.show()

    total_4 = (y_ood == 4).sum()
    total_7 = (y_ood == 7).sum()
    qtd_4_para_9 = ((y_ood == 4) & (y_pred_ood == 9)).sum()
    qtd_7_para_9 = ((y_ood == 7) & (y_pred_ood == 9)).sum()

    print(
        f"\n4 -> 9: {qtd_4_para_9} "
        f"({qtd_4_para_9 / total_4:.2%})"
    )
    print(
        f"7 -> 9: {qtd_7_para_9} "
        f"({qtd_7_para_9 / total_7:.2%})"
    )

    rng = np.random.default_rng(RANDOM_STATE)
    quantidade_amostras = min(10, len(X_ood))
    indices = rng.choice(
        len(X_ood),
        size=quantidade_amostras,
        replace=False,
    )

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flat

    for i, indice in enumerate(indices):
        axes[i].imshow(X_ood[indice].reshape(28, 28), cmap="gray")
        axes[i].set_title(
            f"Real: {y_ood[indice]}\n"
            f"Pred: {y_pred_ood[indice]}"
        )
        axes[i].axis("off")

    for i in range(quantidade_amostras, 10):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    print("""
CONCLUSAO DA FASE 5.2

O SVM foi treinado sem receber nenhuma imagem das classes 4 e 7.
Quando submetido exclusivamente a essas classes, continuou produzindo
previsoes entre as oito classes conhecidas. O experimento demonstra
que um classificador supervisionado nao cria automaticamente uma
categoria desconhecida para entradas fora das classes de treinamento.

As porcentagens 4 -> 9 e 7 -> 9 representam proporcoes de imagens,
e nao probabilidades de confianca. Em aplicacoes reais, cenarios OOD
podem exigir mecanismos adicionais de deteccao de anomalias ou
estimativa de incerteza.
""")

    # ========================================================
    # FASE 5.3 - IMAGENS PROPRIAS
    # ========================================================
    print("\n" + "=" * 70)
    print("FASE 5.3 - INFERENCIA COM IMAGENS MANUSCRITAS PROPRIAS")
    print("=" * 70)

    # Mantem o modelo oficial da Fase 4 intacto.
    # Este possui os mesmos hiperparametros, mas probability=True
    # para permitir o grafico de probabilidades solicitado.
    modelo_svm_prob = SVC(
        C=10,
        kernel="rbf",
        probability=True,
        random_state=RANDOM_STATE,
    )

    print("Treinando SVM probabilistico para a Fase 5.3...")
    inicio = time.time()
    modelo_svm_prob.fit(X_train, y_train)
    tempo_svm_prob = time.time() - inicio
    print(f"Tempo de treinamento: {tempo_svm_prob:.2f} segundos")

    resultados_imagens = executar_fase_5_3(
        modelo_svm_prob,
        PASTA_IMAGENS,
    )

    print("""
CONCLUSAO DA FASE 5.3

As imagens manuscritas proprias representam dados diferentes dos
utilizados originalmente no MNIST. Para torna-las compativeis com
o modelo, foi aplicado um pipeline com escala de cinza, inversao
de cores, bounding box, redimensionamento, centralizacao da massa,
normalizacao para [0, 1] e transformacao para 784 caracteristicas.

A inferencia permite observar a generalizacao do modelo para dados
fora do conjunto MNIST. Diferencas de estilo de escrita, espessura,
posicionamento, escala e qualidade da imagem podem afetar a previsao.
""")

    # ========================================================
    # CONCLUSAO GERAL
    # ========================================================
    print("\n" + "=" * 70)
    print("CONCLUSAO GERAL DO PROJETO")
    print("=" * 70)

    print("""
O projeto demonstrou a aplicacao de Machine Learning para classificacao
de digitos manuscritos utilizando o MNIST. Foram comparados SVM,
Random Forest e MLP com divisao estratificada, normalizacao e conjunto
de teste independente.

A avaliacao utilizou Accuracy, Precision, Recall, F1-Score, matrizes
de confusao e tempo de treinamento, permitindo analisar desempenho e
custo computacional.

Os testes de robustez mostraram o comportamento do classificador diante
de classes ocultadas e de entradas OOD. A inferencia com imagens proprias
complementou a avaliacao ao testar a generalizacao para dados produzidos
fora do dataset original.

Dessa forma, o projeto avaliou nao apenas a capacidade de classificacao,
mas tambem limitacoes, robustez e generalizacao dos modelos.
""")

    return {
        "resultados": resultados,
        "resultados_imagens": resultados_imagens,
        "modelo_svm": modelo_svm,
        "modelo_rf": modelo_rf,
        "modelo_mlp": modelo_mlp,
        "modelo_svm_masked": modelo_svm_masked,
        "modelo_svm_prob": modelo_svm_prob,
    }


if __name__ == "__main__":
    main()

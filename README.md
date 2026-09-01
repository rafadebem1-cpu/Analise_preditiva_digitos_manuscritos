# MNIST VisionLab — Análise Preditiva de Dígitos Manuscritos

> Mini-Projeto Avaliativo — Módulo 2 | Curso de Análise de Dados e Machine Learning  
> **Status:** Em desenvolvimento  
> **Prazo de entrega:** 14/09/2026 às 22h

---

## 1. Sobre o Projeto

Este projeto tem como objetivo desenvolver um **pipeline preditivo multiclasse ponta a ponta** capaz de classificar dígitos manuscritos de 0 a 9 utilizando o dataset **MNIST (`mnist_784`)**.

O projeto compara diferentes abordagens de Machine Learning, desde algoritmos clássicos até redes neurais, avaliando não apenas a performance em um conjunto de teste independente, mas também a capacidade de generalização dos modelos em situações adversas.

Além do benchmark entre três modelos, serão realizados testes de robustez envolvendo:

- Classes que não foram apresentadas durante o treinamento;
- Dados fora da distribuição de treinamento (OOD — Out-of-Distribution);
- Imagens manuscritas produzidas pelo próprio aluno.

---

## 2. Objetivos

### Objetivo geral

Construir, treinar, validar, comparar e estressar modelos de Machine Learning para classificação multiclasse de imagens de dígitos manuscritos.

### Objetivos específicos

- Carregar e explorar o dataset MNIST;
- Compreender a representação vetorial das imagens;
- Avaliar a distribuição das classes;
- Aplicar pré-processamento e normalização;
- Realizar divisão estratificada dos dados;
- Treinar três modelos distintos;
- Ajustar hiperparâmetros;
- Comparar os modelos utilizando métricas de classificação;
- Construir matrizes de confusão;
- Avaliar custo computacional e desempenho;
- Testar modelos com classes ocultadas;
- Realizar inferência OOD;
- Avaliar o conceito de falsa certeza/overconfidence;
- Realizar predição em imagens manuscritas próprias;
- Documentar todo o processo;
- Utilizar Git e GitHub seguindo boas práticas de versionamento.

---

# 3. Dataset

O projeto utilizará o dataset **MNIST**, composto por imagens de dígitos manuscritos de 0 a 9.

Cada imagem possui:

- Dimensão: **28 × 28 pixels**
- Número de características quando vetorizada: **784 features**
- Escala original dos pixels: **0 a 255**
- Classes: **10 dígitos (0 a 9)**

O dataset poderá ser carregado utilizando `scikit-learn`:

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml(
    'mnist_784',
    version=1,
    as_frame=False,
    parser='auto'
)
```

ou utilizando `tensorflow.keras.datasets.mnist`.

---

# 4. Estrutura do Projeto

A estrutura planejada para o repositório será:

```text
mnist-analise-preditiva/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── mnist_analise_preditiva.ipynb
│
├── src/
│   └── (funções e módulos auxiliares, se necessário)
│
├── data/
│   └── (dados locais, somente quando necessário)
│
├── images/
│   └── handwritten/
│       └── (imagens manuscritas próprias)
│
├── results/
│   ├── confusion_matrices/
│   ├── figures/
│   └── reports/
│
└── video/
    └── link_video.txt
```

> A estrutura poderá ser ajustada durante o desenvolvimento, desde que os requisitos do projeto sejam mantidos.

---

# 5. Tecnologias e Bibliotecas

Principais tecnologias previstas:

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow/Keras, caso seja utilizada uma rede neural
- PIL/OpenCV para processamento das imagens próprias
- Git
- GitHub

---

# 6. Fase 1 — Carregamento e Análise Exploratória (EDA)

## 6.1 Carregamento dos dados

- [ ] Importar o dataset MNIST.
- [ ] Separar variáveis preditoras `X` e variável alvo `y`.
- [ ] Exibir as dimensões de `X` e `y`.
- [ ] Verificar tipos dos dados.
- [ ] Verificar valores ausentes, quando aplicável.

## 6.2 Distribuição das classes

- [ ] Verificar quantidade de observações por classe.
- [ ] Apresentar a distribuição dos dígitos de 0 a 9.
- [ ] Avaliar se existe desequilíbrio relevante entre as classes.

## 6.3 Visualização das imagens

Criar uma grade visual de no mínimo **2 × 5 imagens**, contemplando exemplos dos dígitos de 0 a 9.

- [ ] Criar grade com Matplotlib.
- [ ] Exibir imagem.
- [ ] Exibir rótulo correspondente.
- [ ] Interpretar visualmente os dados.

## 6.4 Estrutura dos pixels

As imagens MNIST possuem 28 × 28 pixels, totalizando:

```text
28 × 28 = 784 pixels
```

Cada pixel possui intensidade entre:

```text
0   → preto
255 → branco
```

Para utilização nos modelos de Machine Learning, a matriz bidimensional pode ser transformada em um vetor com 784 características.

Exemplo conceitual:

```text
Imagem 28 × 28
      ↓
Vetorização
      ↓
[784 features]
```

### Interpretação

- A imagem original é uma matriz bidimensional;
- Cada posição representa um pixel;
- Cada pixel possui uma intensidade;
- A transformação para vetor permite que algoritmos tradicionais trabalhem com os dados;
- A normalização dos pixels será aplicada antes do treinamento.

---

# 7. Fase 2 — Pré-processamento e Divisão dos Dados

## 7.1 Divisão dos dados

Será realizada divisão estratificada para preservar a proporção das classes.

Estratégia prevista:

```text
70% → Treinamento
10% → Validação
20% → Teste
```

ou outra configuração tecnicamente justificada.

- [ ] Separar treino.
- [ ] Separar validação.
- [ ] Separar teste.
- [ ] Utilizar `stratify=y`.
- [ ] Garantir que o conjunto de teste permaneça independente.

## 7.2 Normalização

Os pixels serão redimensionados para o intervalo:

```text
[0.0, 1.0]
```

Uma possibilidade é:

```python
X = X / 255.0
```

Também poderá ser utilizado um scaler do Scikit-learn, desde que sua utilização seja tecnicamente justificada.

## 7.3 Justificativa técnica

A normalização é importante porque os pixels possuem originalmente valores entre 0 e 255.

Colocar as características em uma escala comparável pode:

- facilitar a convergência de modelos;
- melhorar o comportamento de algoritmos baseados em distância;
- evitar que escalas maiores dominem determinadas etapas do aprendizado;
- contribuir para um treinamento mais estável.

---

# 8. Fase 3 — Implementação e Treinamento dos Modelos

Serão implementados **três modelos distintos**.

## 8.1 Modelo 1

**Modelo escolhido:** `A DEFINIR`

### Hiperparâmetros

Serão ajustados pelo menos dois hiperparâmetros.

- Hiperparâmetro 1: `A DEFINIR`
- Hiperparâmetro 2: `A DEFINIR`

### Justificativa

`A PREENCHER APÓS A IMPLEMENTAÇÃO`

---

## 8.2 Modelo 2

**Modelo escolhido:** `A DEFINIR`

### Hiperparâmetros

- Hiperparâmetro 1: `A DEFINIR`
- Hiperparâmetro 2: `A DEFINIR`

### Justificativa

`A PREENCHER APÓS A IMPLEMENTAÇÃO`

---

## 8.3 Modelo 3

**Modelo escolhido:** `A DEFINIR`

### Hiperparâmetros

- Hiperparâmetro 1: `A DEFINIR`
- Hiperparâmetro 2: `A DEFINIR`

### Justificativa

`A PREENCHER APÓS A IMPLEMENTAÇÃO`

---

## 8.4 Critérios de comparação

Para cada modelo serão registrados:

- Tempo de treinamento;
- Acurácia;
- Precision;
- Recall;
- F1-Score;
- Comportamento da matriz de confusão;
- Custo computacional;
- Capacidade de generalização.

---

# 9. Fase 4 — Avaliação Comparativa

Os três modelos serão avaliados no conjunto de teste independente.

## 9.1 Métricas

Será construída uma tabela comparativa contendo:

| Modelo | Accuracy | Precision Weighted | Recall Weighted | F1 Weighted | Tempo de Treino |
|---|---:|---:|---:|---:|---:|
| Modelo 1 | A calcular | A calcular | A calcular | A calcular | A calcular |
| Modelo 2 | A calcular | A calcular | A calcular | A calcular | A calcular |
| Modelo 3 | A calcular | A calcular | A calcular | A calcular | A calcular |

## 9.2 Matrizes de confusão

Para cada modelo:

- [ ] Gerar matriz de confusão 10 × 10.
- [ ] Criar heatmap.
- [ ] Identificar classes com maior confusão.
- [ ] Interpretar os erros.

## 9.3 Classification Report

Será utilizado o `classification_report` para analisar:

- Precision;
- Recall;
- F1-score;
- Support.

## 9.4 Conclusão técnica

Ao final da avaliação será respondido:

1. Qual modelo apresentou a melhor performance?
2. Qual modelo apresentou o menor custo computacional?
3. Qual dígito apresentou maior taxa de confusão?
4. Quais pares de dígitos foram mais confundidos?
5. Existe relação entre complexidade do modelo e desempenho?
6. Qual modelo apresenta melhor relação entre desempenho e custo computacional?

> Os resultados serão preenchidos após a execução dos modelos.

---

# 10. Fase 5 — Testes de Robustez e Generalização

A Fase 5 será utilizada para avaliar o comportamento dos modelos em situações diferentes do cenário tradicional de treinamento e teste.

---

# 10.1 Desafio A — Class Masking

Serão selecionadas pelo menos duas classes para serem totalmente removidas do treinamento.

Exemplo:

```text
Classes ocultadas:
4 e 7
```

A definição final das classes será realizada durante o desenvolvimento.

## Procedimentos

- [ ] Selecionar duas classes.
- [ ] Remover completamente essas classes do conjunto de treinamento.
- [ ] Treinar um modelo sem apresentar essas classes durante o ajuste.
- [ ] Registrar quais classes foram ocultadas.
- [ ] Garantir que não exista vazamento dessas classes para o treinamento.

---

# 10.2 Desafio B — Inferência OOD

O modelo treinado sem as classes selecionadas será testado exclusivamente com as classes ocultadas.

Exemplo:

```text
Treinamento:
0 1 2 3 5 6 8 9

Classes não vistas:
4 7
```

## Análises

- [ ] Realizar inferência nas classes ocultadas.
- [ ] Registrar as classes previstas.
- [ ] Gerar matriz de confusão.
- [ ] Analisar quais classes conhecidas foram atribuídas aos dígitos desconhecidos.
- [ ] Avaliar as probabilidades/confiança das previsões, quando disponíveis.
- [ ] Discutir o conceito de falsa certeza.
- [ ] Avaliar o comportamento de overconfidence.

## Questões técnicas

Será analisado:

> Como um classificador multiclasse reage quando recebe uma classe que nunca apareceu durante o treinamento?

Também será discutido que um classificador convencional pode ser obrigado a escolher uma das classes conhecidas, mesmo quando a entrada pertence a uma classe que ele nunca viu.

---

# 10.3 Desafio C — Imagens Manuscritas Próprias

Será criada uma coleção de imagens de dígitos manuscritos produzidas pelo próprio aluno.

As imagens serão armazenadas no diretório:

```text
images/handwritten/
```

## Pipeline de processamento

O pipeline deverá contemplar:

- [ ] Leitura da imagem;
- [ ] Conversão para escala de cinza;
- [ ] Inversão das cores, quando necessária;
- [ ] Identificação do dígito;
- [ ] Recorte/bounding box;
- [ ] Centralização da massa;
- [ ] Redimensionamento para 28 × 28;
- [ ] Normalização para [0.0, 1.0];
- [ ] Adequação ao formato esperado pelo modelo;
- [ ] Predição.

## Visualização

Será apresentada uma visualização contendo:

```text
Imagem processada
        +
Probabilidades das classes
        +
Classe prevista
```

## Análise

Será avaliado:

- Qual dígito o modelo previu;
- Qual foi a probabilidade associada;
- Se houve erro de classificação;
- Quais características da escrita podem ter influenciado o resultado;
- Diferenças entre imagens do MNIST e imagens reais produzidas pelo aluno.

---

# 11. Pipeline Geral do Projeto

O fluxo esperado será:

```text
                 DATASET MNIST
                       │
                       ▼
              CARREGAMENTO DOS DADOS
                       │
                       ▼
                  EDA / VISUALIZAÇÃO
                       │
                       ▼
               PRÉ-PROCESSAMENTO
                       │
                       ▼
             DIVISÃO ESTRATIFICADA
                       │
                       ▼
                  NORMALIZAÇÃO
                       │
                       ▼
             ┌─────────┼─────────┐
             ▼         ▼         ▼
          MODELO 1  MODELO 2  MODELO 3
             │         │         │
             └─────────┼─────────┘
                       ▼
                AVALIAÇÃO
                       │
                       ▼
        COMPARAÇÃO DE PERFORMANCE
                       │
                       ▼
             TESTES DE ROBUSTEZ
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
        CLASS MASKING   OOD   IMAGENS PRÓPRIAS
                       │
                       ▼
              CONCLUSÃO TÉCNICA
```

---

# 12. Controle de Vazamento de Dados (Data Leakage)

Um dos cuidados fundamentais será evitar que informações do conjunto de teste influenciem o treinamento.

Princípios adotados:

- O conjunto de teste será separado antes do treinamento final;
- Transformações que necessitem de `fit` serão ajustadas somente com os dados de treinamento;
- O conjunto de teste será utilizado apenas para avaliação;
- As classes ocultadas no Desafio A não poderão aparecer no treinamento;
- As imagens próprias serão utilizadas somente na etapa de inferência.

---

# 13. Reprodutibilidade

O projeto terá um arquivo:

```text
requirements.txt
```

contendo as dependências utilizadas e suas respectivas versões.

Exemplo de dependências que poderão ser utilizadas:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
keras
pillow
opencv-python
jupyter
```

As versões definitivas serão registradas após a configuração do ambiente.

---

# 14. Git e GitHub

O projeto seguirá uma estratégia de versionamento baseada em branches.

## Branch principal

```text
main
```

A branch `main` conterá a versão final do projeto.

## Branch de desenvolvimento

```text
develop
```

A branch `develop` concentrará o desenvolvimento e os merges das funcionalidades.

## Feature branches

Serão criadas branches específicas para cada etapa.

Exemplo:

```text
develop
│
├── feature/eda
├── feature/preprocessamento
├── feature/modelo-1
├── feature/modelo-2
├── feature/modelo-3
├── feature/avaliacao
├── feature/desafio-ood
├── feature/imagens-proprias
└── feature/documentacao
```

As branches de funcionalidades não deverão ser excluídas após o Pull Request, conforme solicitado no projeto.

---

# 15. Padrão de Commits

Os commits deverão utilizar mensagens curtas, objetivas e no infinitivo.

Exemplos:

```text
implementa carregamento do MNIST
implementa análise exploratória
adiciona visualização das classes
implementa divisão estratificada
implementa normalização dos pixels
adiciona modelo SVM
adiciona modelo Random Forest
adiciona modelo KNN
implementa avaliação dos modelos
adiciona matrizes de confusão
implementa class masking
implementa teste OOD
adiciona pipeline de imagens próprias
atualiza documentação
corrige processamento das imagens
```

---

# 16. Plano de Desenvolvimento

## Etapa 1 — Preparação

- [ ] Criar repositório no GitHub.
- [ ] Criar branch `develop`.
- [ ] Criar estrutura inicial de diretórios.
- [ ] Criar `README.md`.
- [ ] Criar `requirements.txt`.
- [ ] Configurar ambiente Python.

## Etapa 2 — EDA

- [ ] Carregar MNIST.
- [ ] Verificar dimensões.
- [ ] Verificar distribuição das classes.
- [ ] Visualizar imagens.
- [ ] Documentar interpretação dos pixels.

## Etapa 3 — Pré-processamento

- [ ] Divisão estratificada.
- [ ] Normalização.
- [ ] Preparação dos dados para os modelos.

## Etapa 4 — Modelagem

- [ ] Escolher Modelo 1.
- [ ] Escolher Modelo 2.
- [ ] Escolher Modelo 3.
- [ ] Definir hiperparâmetros.
- [ ] Treinar modelos.
- [ ] Registrar tempos de treinamento.

## Etapa 5 — Avaliação

- [ ] Accuracy.
- [ ] Precision.
- [ ] Recall.
- [ ] F1-score.
- [ ] Classification report.
- [ ] Matrizes de confusão.
- [ ] Comparação dos modelos.
- [ ] Conclusões técnicas.

## Etapa 6 — Robustez

- [ ] Class masking.
- [ ] Teste OOD.
- [ ] Matriz de confusão OOD.
- [ ] Análise de falsa certeza.

## Etapa 7 — Imagens próprias

- [ ] Produzir imagens manuscritas.
- [ ] Criar pipeline de processamento.
- [ ] Redimensionar.
- [ ] Centralizar.
- [ ] Normalizar.
- [ ] Realizar predições.
- [ ] Visualizar probabilidades.

## Etapa 8 — Finalização

- [ ] Revisar código.
- [ ] Revisar README.
- [ ] Atualizar requirements.txt.
- [ ] Verificar caminhos relativos.
- [ ] Testar execução do projeto.
- [ ] Revisar branches.
- [ ] Realizar merge final para `main`.
- [ ] Gravar vídeo.
- [ ] Disponibilizar vídeo no Google Drive.
- [ ] Inserir links finais na submissão do AVA.

---

# 17. Requisitos de Execução

Após clonar o projeto:

```bash
git clone URL_DO_REPOSITORIO
cd mnist-analise-preditiva
```

Criar ambiente virtual:

```bash
python -m venv .venv
```

Ativar no Windows:

```bash
.venv\Scripts\activate
```

Instalar dependências:

```bash
pip install -r requirements.txt
```

Executar o Jupyter:

```bash
jupyter notebook
```

Abrir o notebook:

```text
notebooks/mnist_analise_preditiva.ipynb
```

> Os comandos definitivos poderão ser ajustados conforme a configuração final do projeto.

---

# 18. Caminhos Relativos

O projeto não deverá utilizar caminhos absolutos específicos da máquina do desenvolvedor.

Evitar:

```python
"C:/Users/Rafael/Documents/projeto/imagem.png"
```

Preferir caminhos relativos, por exemplo:

```python
"images/handwritten/imagem.png"
```

ou utilizando `pathlib`.

Isso garante maior reprodutibilidade e permite que o projeto seja executado em outras máquinas.

---

# 19. Vídeo de Apresentação

Será produzido um vídeo com duração máxima de **10 minutos**, contendo:

1. Objetivo do sistema;
2. Demonstração de funcionamento;
3. Como executar o projeto;
4. Organização das tarefas antes do desenvolvimento;
5. Branches criadas e objetivos;
6. Melhorias que poderiam ser implementadas;
7. Justificativa das escolhas técnicas;
8. Resultados obtidos;
9. Análise dos desafios de robustez;
10. Conclusão.

O vídeo deverá:

- Mostrar o rosto do aluno;
- Possuir boa iluminação;
- Ser disponibilizado no Google Drive;
- Ter permissão de leitura para qualquer pessoa com o link.

### Link do vídeo

`A PREENCHER NA VERSÃO FINAL`

---

# 20. Melhorias Futuras

Possíveis evoluções do projeto:

- Utilização de CNNs (Convolutional Neural Networks);
- Data augmentation;
- Técnicas específicas de detecção OOD;
- Calibração das probabilidades;
- Análise de confiança;
- Explainable AI (XAI);
- Comparação com modelos de Deep Learning mais avançados;
- Interface web utilizando Streamlit;
- API para classificação de imagens;
- Monitoramento de desempenho do modelo;
- Testes com diferentes tipos de escrita;
- Avaliação de robustez contra ruído e transformações nas imagens.

---

# 21. Conclusões

`A SER PREENCHIDO APÓS A EXECUÇÃO DO PROJETO`

Nesta seção serão consolidados:

- Melhor modelo;
- Principais métricas;
- Principais confusões;
- Custo computacional;
- Resultado do class masking;
- Resultado do teste OOD;
- Resultado das imagens próprias;
- Principais limitações;
- Melhorias futuras.

---

# 22. Checklist Final de Entrega

## Aplicação

- [ ] Dataset MNIST carregado.
- [ ] Dimensões apresentadas.
- [ ] Distribuição das classes apresentada.
- [ ] Grade 2 × 5 de imagens criada.
- [ ] Estrutura dos pixels explicada.
- [ ] Dados divididos de forma estratificada.
- [ ] Dados normalizados.
- [ ] Três modelos implementados.
- [ ] Pelo menos dois hiperparâmetros ajustados por modelo.
- [ ] Justificativas técnicas registradas.
- [ ] Matrizes de confusão dos três modelos.
- [ ] Accuracy calculada.
- [ ] Precision calculada.
- [ ] Recall calculado.
- [ ] F1-score calculado.
- [ ] Comparação dos modelos realizada.
- [ ] Análise de custo computacional realizada.
- [ ] Class masking implementado.
- [ ] Teste OOD implementado.
- [ ] Matriz de confusão OOD criada.
- [ ] Discussão sobre overconfidence realizada.
- [ ] Imagens manuscritas próprias utilizadas.
- [ ] Pipeline de processamento das imagens implementado.
- [ ] Imagens redimensionadas para 28 × 28.
- [ ] Centralização implementada.
- [ ] Normalização implementada.
- [ ] Probabilidades das previsões apresentadas.

## GitHub

- [ ] Repositório público.
- [ ] Branch `develop`.
- [ ] Feature branches.
- [ ] Commits claros e objetivos.
- [ ] Pull Requests utilizados.
- [ ] Branches de funcionalidades mantidas.
- [ ] Código final mergeado em `main`.
- [ ] `README.md` completo.
- [ ] `requirements.txt` atualizado.
- [ ] Caminhos relativos utilizados.
- [ ] Imagens próprias organizadas no repositório.

## Vídeo

- [ ] Vídeo gravado.
- [ ] Duração máxima de 10 minutos.
- [ ] Rosto visível.
- [ ] Boa iluminação.
- [ ] Demonstração do sistema.
- [ ] Explicação das escolhas técnicas.
- [ ] Explicação da organização do projeto.
- [ ] Explicação das branches.
- [ ] Discussão das melhorias futuras.
- [ ] Vídeo disponibilizado no Google Drive.
- [ ] Permissão "qualquer pessoa com o link" configurada.
- [ ] Link submetido no AVA.

---

# 23. Uso de Inteligência Artificial

De acordo com as regras do projeto, ferramentas de Inteligência Artificial serão utilizadas apenas como **suporte ao aprendizado**, especialmente para:

- esclarecimento de conceitos;
- explicação de bibliotecas e métodos;
- auxílio na interpretação da documentação;
- depuração de erros de sintaxe;
- discussão de alternativas técnicas;
- apoio à compreensão dos resultados.

O código final deverá ser **compreendido e desenvolvido pelo aluno**, evitando geração integral e não supervisionada da solução.

Durante o desenvolvimento, cada etapa será construída e validada gradativamente.

---

# 24. Autor

**Aluno:** Rafael de Oliveira  
**Curso:** Análise de Dados e Machine Learning  
**Projeto:** Mini-Projeto Avaliativo — Módulo 2  
**Ano:** 2026

---

## Status do Projeto

🚧 **Em desenvolvimento**

As seções marcadas como `A DEFINIR` ou `A PREENCHER` serão atualizadas à medida que o projeto for desenvolvido e os resultados forem obtidos.

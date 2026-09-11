# Classificação de Dígitos Manuscritos (MNIST)

Este repositório contém um **Mini-Projeto Avaliativo de Machine Learning** focado na classificação de dígitos manuscritos utilizando o dataset **MNIST**. O projeto aborda desde a Análise Exploratória de Dados (EDA) até o treinamento e avaliação comparativa de modelos supervisionados (SVM, Random Forest e MLP), análises de robustez com *Class Masking* / *Out-of-Distribution* (OOD) e inferência em imagens manuscritas próprias.

---

## 📋 Estrutura do Projeto

```text
.
├── mini_projeto_final.py    # Código principal do projeto
├── requirements.txt         # Dependências do projeto
├── data/
│   └── handwritten/         # Pasta para imagens manuscritas próprias (ex: digito_5.png)
└── README.md                # Documentação do repositório
```

---

## 🛠️ Pré-requisitos e Instalação

### 1. Clonar o Repositório
```bash


git clone https://github.com/rafadebem1-cpu/Analise_preditiva_digitos_manuscritos.git

cd Analise_preditiva_digitos_manuscritos.git
```

### 2. Criar e Ativar um Ambiente Virtual (Opcional, mas Recomendado)
* **Linux/macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
* **Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

### 3. Instalar Dependências
Certifique-se de ter o `pip` atualizado e instale todas as bibliotecas necessárias:
```bash
python -m pip install -r requirements.txt
```

#### Conteúdo do `requirements.txt`:
```text
ipykernel
numpy
matplotlib
tensorflow
scikit-learn
seaborn
keras
pandas
```

---

## 🚀 Como Executar o Projeto

1. **(Opcional) Adicionar Imagens Próprias**:
   Coloque imagens de dígitos manuscritos (nos formatos `.png`, `.jpg` ou `.jpeg`) na pasta `data/handwritten/`. 
   > **Dica:** Nomeie os arquivos incluindo o dígito correto para validação automática (exemplo: `digito_5.png`, `teste_7.jpg`).

2. **Rodar o Script Principal**:
   Execute o seguinte comando no terminal:
   ```bash
   python mini_projeto_final.py
   ```

---

## 📌 Etapas do Pipeline

O script `mini_projeto_final.py` é dividido em 5 fases principais:

1. **Fase 1 - Carregamento e Análise Exploratória (EDA)**
   - Download/Carregamento do dataset MNIST (`70.000` imagens `28x28`).
   - Exibição da distribuição das classes e amostras visuais em grade $2 	imes 5$.

2. **Fase 2 - Pré-Processamento e Divisão dos Dados**
   - Divisão estratificada dos dados: Treino ($70\%$), Validação ($10\%$) e Teste ($20\%$).
   - Normalização dos valores dos pixels de $[0, 255]$ para o intervalo $[0, 1]$.

3. **Fase 3 - Treinamento dos Modelos**
   - Treinamento e ajuste dos modelos no conjunto de treino e validação:
     - **SVM** (Kernel RBF, $C=10$)
     - **Random Forest** ($150$ árvores, profundidade máxima $20$)
     - **MLP** (Rede Neural Multicamadas: $128 	imes 64$ neurônios)

4. **Fase 4 - Avaliação no Conjunto de Teste**
   - Avaliação detalhada utilizando métricas de *Accuracy*, *Precision*, *Recall* e *F1-Score* (weighted).
   - Geração de Matrizes de Confusão e identificação dos principais erros por classe.

5. **Fase 5 - Testes de Robustez e Inferência Real**
   - **5.1 Class Masking**: Treinamento do SVM com ocultação deliberada das classes `4` e `7`.
   - **5.2 Inferencia Out-of-Distribution (OOD)**: Avaliação do comportamento do modelo oculto ao receber apenas imagens das classes `4` e `7`.
   - **5.3 Imagens Manuscritas Próprias**: Pré-processamento customizado (escala de cinza, inversão, *crop*, redimensionamento, centralização de massa) e classificação de imagens externas enviadas pelo usuário.

---

## 📊 Principais Resultados

- **Acurácia nos Modelos (Teste)**: Comparativo do desempenho entre SVM, Random Forest e MLP.
- **Comportamento OOD**: Demonstração empírica de que modelos supervisionados tradicionais tendem a classificar amostras fora do seu domínio em classes conhecidas com similaridade morfológica (ex.: dígitos $4 	o 9$ e $7 	o 9$).
- **Pipeline de Pré-processamento**: Tratamento e normalização robustos para aproximar fotos/desenhos externos ao padrão de matrizes $28 	imes 28$ do MNIST.
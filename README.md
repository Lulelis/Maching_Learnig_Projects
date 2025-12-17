### **Projeto de Machine Learning** – Classificação Supervisionada no Titanic (_Survived vs Not Survived)_

![Titanic_dataset](https://raw.githubusercontent.com/Masterx-AI/Project_Titanic_Survival_Prediction_/main/titanic.jpg)

### 🚢 Contextualização & Objetivos do Projeto:

> **Observação**: Importante ressaltar que esse projeto foi realizado como projeto prático da disciplina de Machine Learning via PUC-RS (_todos os direitos reservados_), (dataset foi adaptado e disponibilizado para cumprimento dos requistos técnicos do projeto)
> no entanto, mesmo sendo uma versão adaptada para os estudantes realizarem o projeto, é um conjunto de dados provenientes do Kaggle (*https://www.kaggle.com/datasets/yasserh/titanic-dataset*)

- _<u>Dados as tarefas que são mais importantes para o desenvolvimento e sucesso de uma solução baseada em algoritmos de Machine Learning, estão</u>_: a coleta e preparação dos dados. Na Etapa de coleta, reunimos os dados necessários para a construção da solução. Já, na etapa de preparação, é analisado, filtrado e preparado os dados para aplicação desses algoritmos.

---

- Dessa forma, os objetivos principais deste de projeto podem ser elencados em:

  - Análise, limpeza, filtragem & Tratamento dos Dados (EDA)
  - Seleção das **_feautures_** que são mais relevantes para a compor e dar entrada aos algoritmos de **Machine Learning**/Preparação dos Dados
  - Utilização dos Dados preparados para construção de um classificador Binário de Predição em um problema de Classificação (Taxa de Predição entre passageiros que: Sobreviveram e Não sobreviveram)
  - Balanceamento do Dataset
  - Descrição dos Algoritmos de Aprendizado Supervisionado para resolução do problema
  - Separação do subconjunto em: Treino e Teste (Estratificado) utilizando validação Cruzada
  - Comparação dos resultados utilizando as medidas: Accuracy(Acurácia), Precision(Precisão), Recall (Sensibilidade) e F-Mesaure (Coeficiente F)
  - Matriz de confusão de cada experimento
  - Conclusão e hipótese comentada

#### Ferramentas e Biblioteca que foram Utilizadas:

- <img 
    alt="VS Code"
    title="IDE Visual Studio Code"
    width="24px"
    style="vertical-align: middle; margin-right: 6px;"
    src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/vscode/vscode-original.svg"
  /> **Visual Studio Code**
- <img 
    alt="Python"
    title="Linguagem Python Intermediária"
    width="24px"
    style="vertical-align: middle; margin-right: 6px;"
    src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg"
  /> **Python 3.12.12**

  - Bibliotecas:

    - <img
        alt="Pandas"
        title="Biblioteca Pandas"
        width="30px"
        style="vertical-align: middle; margin-right 6px;"
        src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/pandas/pandas-original-wordmark.svg"
        />
    - <img
        alt="Numpy"
        title="Biblioteca Numpy"
        width="30px"
        style="vertical-align: middle; margin-right 6px;"
        src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/numpy/numpy-original-wordmark.svg"
        />
    - <img
        alt="ScikitLearn"
        title="Biblioteca ScikitLearn"
        width="30px"
        style="vertical-align: middle; margin-right 6px;"
        src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/scikitlearn/scikitlearn-original.svg"
        />
    - <img
        alt="Seaborn"
        title="Biblioteca Seaborn"
        width="26px"
        style="vertical-align: middle; margin-right: 6px;"
        src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg"
      /> **Seaborn**

    - <img
        alt="Matplotlib"
        title="Biblioteca Matplotlib"
        width="30px"
        style="vertical-align: middle; margin-right: 6px;"
        src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/matplotlib/matplotlib-original-wordmark.svg"
      /> **Matplotlib**

    - <img
        alt="Plotly"
        title="Biblioteca Plotly Express"
        width="35px"
        style="vertical-align: middle; margin-right: 6px;"
        src="https://upload.wikimedia.org/wikipedia/commons/8/8a/Plotly-logo.png"
      /> **Plotly Express**

---

#### 🎲 **EDA**- Análise Exploratória de Dados

_Observação_: Toda as análises coluna à coluna como o tratamento para enriquecimento dos dados, tratamento de dados faltantes bem como ao seu tipo e visualizações e etc.. Foram comentadas passo a passo ao lonog do script do notebook do código!

- O dataset Modificado possui as seguintes colunas:

* Tamanho/RangeIndex do dataset: 1069

**Análise Coluna a Coluna**
Coluna|Situação|Tipo de Dado
-----|-----------|--------------|
PassengerId| Identificador Único| int64  
Survived|Alvo (0/1)|bool  
Pclass|Classe do passageiro|float64
Name|Texto livre| object
Sex|Sexo|object
SibSp|Nº de irmãos/cônjuges a bordo|int64  
Parch|Nº de pais/filhos a bordo | int64  
Ticket|Texto livre | object
Fare|Tarifa paga|float64
Cabin|77% nulos|object
Embarked|Porto de embarque|object
day; month; year; time|Datas|int64  
cost; budget|80–90% nulos|float64
age|Idade (20% nulos)|object

> 💹 **Etapas GERAIS cumpridas durante a EDA**:

1. Visualização dos dados para cada coluna
2. Formatação de para coluna
3. Remoção das linhas duplicadas + tratamento de dados ausentes e técnicas empregadas
4. Transformação das colunas categóricas para numéricas

---

- **Subetapas por ordem de empregação, de acordo com o notebook**:

  - Análise e escolha dos atributos que serão necessários na entrada dos algoritmos:
  - Discussão comentada em formato markdown nas células do notebook a respeito da escolha de Atributos (Colunas) e _feature engenerring & enriquecimento dos dados_ das colunas nas quas pensei que seriam mais relevantes para entrada dos modelos e obter métricas de classificação mais altas...

    - ETAPAS CUMPRIDAS:

      - 1. Visualização dos dados
      - 2. Análise sobre possiveis relação encontradas
      - 3. Escolha das colunas mais relevântes mais justificativa comentada

- **Preenchimento de dados faltantes**:

  - ETAPAS CUMPRIDAS:

    - 1. Gerar o dataset sem as colunas com dados faltantes
    - 2. Aplicar o KNN Imputer/hot deck nas colunas + realizações de comparações gráficas de dispersão (box_plot) entre métodos de moda e o algoritmo empregado
    - 3. Cálcular um atributo estatístico (média, moda, mediana, etc), no dataset original, das colunas com dados faltantes
    - 4. Preencher os valores das linhas com dados faltantes com o atributo estatístico (se necessário) referente ao seu grupo

- **Escala dos atributos**:Agora, será necessário reescalar os dados, para que os algoritmos consigam aprender as relações entre os dados sem muito ruído e melhorar a classificação.Obs: Para realizar essa etapa, é necessário verificar e tratar colunas com outliers. - ETAPAS CUMPRDAS - Verificar quais dados possuem outliers e tratar de acordo - Reescalar os valores

- **Checklist-Desta Etapa**:
  - Remoção das colunas que não acrescentam informações úteis;
  - Todas as colunas estão com a sua representação unificada;
  - Todas as colunas estão no formato numérico;
  - Todas as colunas estão com dados válidos (sem dados falantes);
  - Todas as colunas estão reescaladas;
  - Coluna alvo balanceada.

> 🤖 Aplicação e Validação de Algoritmos de Machine Learning _Etapas Gerais Cumpridas_:

- Seleção e treinamento de ao menos 4 algoritmos de classificação para para o dataset;
  -Ciclo de Treinamento, validação e teste do modelo dividido em: Exploração e ajuste de hiperparâmetros (para os modelos escolhidos), visando o melhor resultado do classificador.
  -Validação dos modelos usando CROSS-VALIDATION;
- Utilização da análise da matriz de confusão e aplicação de métricas de avaliação (**_Accuracy, Precision, Recall & F-Measure_**);
- Análise descritiva e comentada dos resultados, dividido em: Análise dos resultados de cada algoritmo e comparação entre os algoritmos.
- Comentários gerais sobre o desempenho do classificador, mencuionando acertos e discutiondo possíveis razões para os erros;

- _Escolha dos algoritmos,utilização de técnicas de modelagem (Comitê de Aprendizes (Ensemble Classifiers) testado no algoritmo Decision Tree), para melhorar o desempenho dos modelos e aplicação do GridSearch exploração dos hiper-parâmetros_;

> Importante acrescentar, que:

Além dos modelos tradicionais, utilizei também técnicas de Comitê de Aprendizes
(Ensemble Learning), como o BaggingClassifier. Esse tipo de abordagem combina
múltiplos aprendizes fracos,no caso a árvores de decisão, para reduzir variância
e aumentar a estabilidade do modelo. O objetivo do comitê não foi validar
hiperparâmetros, mas sim melhorar o desempenho geral do classificador por meio da
agregação de vários modelos independentes.

- Criação e testagem dos modelos que foram utilizados:
  - **_Árvore de decisão_**,
  - **_MLPClassifier_**,
  - **_KNN_**,
  - **_GradienteBoostingClassifier_**, (_Gradient Boosting é um algoritmo de Comitê de Aprendizes (Ensemble Learning) baseado em boosting. Ele treina vários modelos fracos de forma sequencial, onde cada modelo corrige os erros do anterior. Diferente do Bagging, ele não faz amostragem com reposição e utiliza o gradiente do erro para ajustar os modelos seguintes_),
  - **_Naive_Bayes_**,
  - **_Random Forest_** e
  - **_Dummy_** para baseline

#### Modelos Selecionados para Hiperparametrização

Após a etapa inicial de testes, alguns modelos foram descartados por apresentarem
desempenho inferior ou por não se adequarem bem ao problema. Assim, para a etapa de
busca pelos melhores hiperparâmetros (GridSearchCV), foram selecionados os seguintes
modelos:

- **Decision Tree Classifier**
- **Random Forest Classifier**
- **MLPClassifier (Rede Neural Artificial)**
- **K-Nearest Neighbors (KNN)**
- **GradientBoostingClassifier**

Além desses, foi incluído também o:

- **DummyClassifier (Baseline)**  
  _O DummyClassifier é utilizado como referência mínima de desempenho. Ele não aprende
  padrões dos dados; apenas gera previsões constantes ou aleatórias. Por esse motivo,
  não possui hiperparâmetros relevantes para ajuste via GridSearch. Seu papel é
  demonstrar o desempenho mínimo esperado, permitindo avaliar se os modelos reais estão
  de fato aprendendo e superando o baseline._

#### Execução do Treinamento e Validação usando Cross-Validation (_Conjunto de dados pequeno 1069 entradas, após toda etapa de EDA, reduzido a 890 entradas_):

- Mesmo que o GridSearchCV use internamete a validação cruzada (por exemplo, cv=5) para escolher os melhores hiperparâmetrosm, ainda não foi feito uma validação cruzada final comparando todos os modelos com as métricas necessárias:

  metrics = {
  'accuracy': 'Accuracy',
  'f1': 'F1-Score',
  'roc_auc': 'ROC-AUC',
  'precision': 'Precision',
  'recall': 'Recall'
  }

**_DESEMPENHO DOS MODELOS (KFold - 10 splits)_**:
| Modelo | Accuracy | F1-Score | ROC-AUC | Precision | Recall | Tempo_Treino (s) |
|-------------------|--------------------|--------------------|--------------------|--------------------|--------------------|------------------|
| Gradient Boosting | 0.8371 ± 0.0435 | 0.7829 ± 0.0539 | 0.8864 ± 0.0403 | 0.7974 ± 0.0798 | 0.7784 ± 0.0808 | 0.39 |
| Random Forest | 0.8286 ± 0.0423 | 0.7672 ± 0.0501 | 0.8934 ± 0.0421 | 0.8008 ± 0.0714 | 0.7481 ± 0.0939 | 0.36 |
| Decision Tree | 0.8175 ± 0.0546 | 0.7499 ± 0.0677 | 0.8743 ± 0.0490 | 0.7976 ± 0.1094 | 0.7259 ± 0.1113 | 0.00 |
| MLP | 0.8216 ± 0.0348 | 0.7486 ± 0.0448 | 0.8718 ± 0.0368 | 0.8074 ± 0.0823 | 0.7063 ± 0.0707 | 1.81 |
| KNN | 0.7964 ± 0.0581 | 0.7316 ± 0.0638 | 0.8782 ± 0.0510 | 0.7458 ± 0.0933 | 0.7303 ± 0.0925 | 0.00 |
| Dummy | 0.5435 ± 0.0586 | 0.3442 ± 0.0969 | 0.5012 ± 0.0645 | 0.3874 ± 0.1350 | 0.3165 ± 0.0821 | 0.04 |

### > 🎉 Conclusão:

#### > De acordo com a execução dos meus modelos, após validação cruzada usando o K-fold (por seu um conjunto de dados pequeno);

> Após a execução dos modelos e a validação cruzada utilizando **K-Fold com 10 divisões**, foi possível comparar o desempenho médio de cada algoritmo com base nas principais métricas de classificação. Os resultados obtidos para esse _famoso_ dataset de treinamento de Machine Learning do Kaggle, demonstram diferenças (dado a forma que executei e as decisões que escolhi tomar), serem relevantes entre os modelos, tanto em desempenho quanto em estabilidade.

#### **_1._** **Melhor desempenho Geral: Gradiente Boosting**

O Gradient Boosting apresentou o melhor equilíbrio entre as métricas avaliadas, alcançando:

- **Acurácia**: 83.71%
- **F1-Score**: 78.29%
- **Recall** (_Sensibilidade_): 77.84%
- **ROC-AUC**: 88.64%

Tais valores expressos pelos métricas de cálculo, idnicam que o modelo conseguiu identificar corretamente tanto sobreviventes quanto não sobreviventres, mantendo boa capacidade de generelização (na medida do possível). Entre todos os algoritmos testatos, foi o mais consistente e robusto.

#### **_2_**. **Segundo melhor modelo: Random Forest**

O Random Forest apresentou desempenho muito próximo ao Gradient Boosting, com:

- **Acurácia**: 82.86%
- **F1-Score**: 76.72%
- **ROC-AUC**: 89.34% (melhor entre todos os modelos)
- **Precisão**: 80.08%

Apesar de ligeiramente inferior em F1-Score e Recall, o Random Forest superou o Gradiente Boosting (dado todas as condições de testagem que realizei) em **Precisão** e **ROC-AUC**, mostrando excelente capacidade de discriminação entre as classes

#### **_3_** **Modelos Intermediários**

Os modelos **_Decision Tree_**, **_MLPClassifier_** e **_KNN_** apresentaram resultados aceitáveis:

- **Decision Tree**: ~81.75% de acurácia
- **MLPClassifier**: ~82.16%
- **KNN**: ~79.64%
  Embora inferiores aos ensembles, ainda assim, superaram amplamente o baseline, demonstrando que extraíram padrões relevantes do dataset.

#### **_4. Baseline_**: **Dummy Classifier**

O DummyClassifier obteve apenas:

- **Acurácia**: 54.35%
- **F1-Score**: 34.42%
  Esse desempenho confirma que os modelos supervisionados realmente aprenderam padrões significativos, já que todos superaram o baseline com ampla margem

#### **5. Métricas mais relevantes para o problema**

Como o OBJETIVO é prever quem sobreviveu ao naufrágio, as métricas mais importantes são:

- **Recall (Sensibilidade)**: minimizar falsos negativos (não prever um sobrevivente)

- **ROC-AUC**: avaliar a capacidade de separação entre as classes
  Errar um sobrevivente é mais crítico do que errar um não sobrevivente, o que torna o Recall uma métrica essencial. O Gradient Boosting apresentou o melhor equilíbrio entre Recall e F1-Score, enquanto o Random Forest se destacou no ROC-AUC.

#### IMPORTANTE: 🙋🏻‍♂️ **6. Considerações sobre custo computacional**

O Gradient Boosting apresentou o maior tempo de execução durante a busca pelos melhores hiperparâmetros, levando aproximadamente 5 minutos e 41 segundos em máquina local. Apesar disso, o ganho de desempenho justifica o custo computacional em cenários onde a precisão é prioritária.

#### Plotagem geradas:

## 🧩 **Limitações do Estudo**

É importante ressalta que: este projeto representa minha primeira experiência prática com algoritmos de Aprendizado de Máquina envolvendo técnicas de Comitês de Aprendizes (Ensemble Learning),
como Bagging, e Gradient Boosting. Embora os resultados tenham sido satisfatórios, algumas limitações devem ser consideradas:

1. **Tamanho reduzido do dataset**  
   O conjunto de dados do Titanic é relativamente pequeno, o que limita a capacidade
   dos modelos de capturar padrões mais complexos e aumenta a variabilidade entre os folds.

2. **Dependência de pré-processamento manual**  
   Algumas decisões de limpeza, transformação e engenharia de atributos foram feitas
   manualmente. Outras abordagens poderiam gerar features mais informativas.

3. **Exploração limitada de hiperparâmetros**  
   Apesar do uso de GridSearchCV, a busca foi restrita a um conjunto específico de
   hiperparâmetros devido ao custo computacional em máquina local.

4. **Pouca experimentação com outros ensembles avançados**

   Modelos como XGBoost, LightGBM e CatBoost não foram explorados, embora sejam
   referências modernas em boosting.

5. **Custo computacional**  
   O Gradient Boosting, por exemplo, levou mais de 5 minutos para encontrar a melhor
   combinação de hiperparâmetros, o que limita experimentos mais amplos.

## 🚀 Trabalhos Futuros

Com base nas limitações identificadas, algumas melhorias e extensões podem ser
implementadas em versões futuras deste projeto:

1. **Explorar ensembles mais modernos**  
   Incluir algoritmos como XGBoost, LightGBM e CatBoost, que oferecem melhor
   desempenho e menor tempo de treinamento.

2. **Aprimorar a engenharia de atributos**  
   Criar novas variáveis derivadas (ex.: tamanho da família, título social,
   agrupamento de idades) para enriquecer o poder preditivo dos modelos.

3. **Aplicar técnicas de balanceamento**  
   Métodos como SMOTE ou class weights podem melhorar o Recall para a classe
   minoritária (sobreviventes).

4. **Automatizar o pipeline**  
   Utilizar ferramentas como `Pipeline` e `ColumnTransformer` para padronizar
   pré-processamento e reduzir risco de vazamento de dados.

5. **Avaliar interpretabilidade**  
   Aplicar SHAP ou LIME para entender melhor a contribuição de cada feature nos
   modelos ensemble.

6. **Comparar desempenho em ambientes mais robustos**  
   Executar a hiperparametrização em máquinas mais potentes ou ambientes em nuvem
   para expandir a busca de parâmetros.

### Autor:

**Lucas Lelis**

- _Projeto Prático ***ADAPTADO*** da Disciplina de Machine Learning-PUC-RS_
- _todos os direitos reservados_

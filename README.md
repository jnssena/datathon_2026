# Datathon - Pós Tech Data Analytics - FIAP

## Objetivo do Projeto
Desenvolver uma análise de data analytics e um modelo de Machine Learning para apoiar a Associação Passos Mágicos a identificar, com antecedência, quais alunos correm risco de entrar em defasagem escolar no ciclo seguinte. O sistema foi deployado via Streamlit como uma aplicação preditiva educacional, contendo um módulo de avaliação individual, uma lista de priorização da turma e um dashboard com os principais insights sobre os dados.

## Ferramentas
- Python
- Jupyter Notebook
- Pandas e NumPy
- Matplotlib e Seaborn
- Scikit-learn
- Streamlit

## Metodologia
### 1. Coleta dos Dados:
Os dados foram obtidos a partir de três bases anuais (BASE DE DADOS PEDE 2022, 2023 e 2024), geradas a partir dos arquivos originais em Excel pelo script `exportar_csv.py` e disponíveis no próprio repositório do GitHub, contendo os indicadores da Pesquisa Extensiva do Desenvolvimento Educacional (PEDE) de cada aluno atendido pela associação. O dicionário dos dados está contido no arquivo PEDE_Dicionario de Dados.pdf. dentro da pasta Arquivos_Base

### 2. Análise Exploratória:
Foram realizadas análises de distribuição da defasagem e do INDE ao longo dos três anos, estatísticas descritivas dos indicadores (IAN, IDA, IEG, IAA, IPS, IPP e IPV) e geração de um gráfico de correlação entre eles. As três bases anuais foram consolidadas em um único painel no formato aluno x ano, com padronização de fase, gênero, instituição de ensino e Pedra entre os anos. As variáveis categóricas do modelo (gênero e instituição) foram codificadas via Label Encoding, com encoders individuais por coluna para permitir a inversão da transformação durante a predição.

O Random Forest foi selecionado como modelo final, atingindo AUC de 0,89 e capturando 80% dos alunos que de fato entrariam em defasagem ao priorizar os 25% de maior risco.

### 3. Pipeline de Machine Learning:
O modelo final foi organizado em um `Pipeline` composto por três etapas:
1. **SimpleImputer** - preenchimento dos indicadores não avaliados naquele ano
2. **StandardScaler** - normalização das features
3. **RandomForestClassifier** - classificação binária de risco de defasagem

A variável alvo foi construída de forma supervisionada, comparando a defasagem do aluno no ano t com a do ano t+1: entrou em risco quem piorou a defasagem e permaneceu defasado no ciclo seguinte. O treino usou as transições 2022 → 2023 e o teste usou as transições 2023 → 2024, simulando o uso real do modelo (treinar com o histórico e prever a turma atual).

### 4. Deploy via Streamlit:
A aplicação foi deployada no Streamlit com três abas principais:

- **Avaliar Aluno**: formulário lateral com os dados do aluno (perfil, fase e indicadores), exibindo a probabilidade de risco, o INDE e o IAN estimados e a matriz de priorização individual.
- **Priorizar Turma**: lista de todos os alunos do último ano ordenados por probabilidade de risco, com slider de capacidade de atendimento e exportação da lista em CSV.
- **Dashboard**: painel com KPIs e gráficos - evolução do INDE, composição por Pedra, nível de defasagem por ano e relação entre engajamento e aprendizagem.

## Links
- **Streamlit**: https://datathon2026-de3z548wekpb6hdlrbbenj.streamlit.app/
- **Apresentação do Projeto**: <link do vídeo aqui>

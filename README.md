# Sistema Neuro-Fuzzy para Triagem de Sintomas

**Integrantes**
- Diogo Buzatto (dbuzatto)
- Lucas Ferreira (lucasfeva)

Descrição
-	Projeto que implementa um sistema neuro-fuzzy inspirado em ANFIS para triagem de sintomas e classificação de gravidade em três classes: Autocuidado, Consulta Médica e Emergência. O repositório contém código para preparar múltiplos datasets, criar labels de severidade, treinar um modelo aproximado de ANFIS e avaliar os resultados.

Principais arquivos
-	`main.py` — Pipeline principal: carregamento, pré-processamento, criação de labels de gravidade, divisão dos dados, treino e avaliação.
-	`api.py` — API (FastAPI) para expor o modelo/serviços (iniciar com `uvicorn api:app --reload`).
-	`requirements.txt` — Dependências do projeto.
-	`datasets/` — Pasta com CSVs usados (ver seção Datasets).

Funcionalidades
-	Carregamento de múltiplos CSVs e fusão automática quando compatíveis.
-	Pré-processamento de features (binárias, idade, gênero, pressão arterial, colesterol).
-	Geração de labels de severidade (0=Autocuidado, 1=Consulta Médica, 2=Emergência) com regras clínicas heurísticas.
-	Modelo aproximado de ANFIS implementado com camadas Keras.
-	Treinamento com callbacks (EarlyStopping, ReduceLROnPlateau) e avaliação com matriz de confusão e relatório de classificação.

Requisitos
-	Python 3.8+ (recomendado 3.9/3.10)
-	Recomenda-se GPU para treinos maiores, mas funciona em CPU.

Instalação
```bash
# criar e ativar venv (bash / zsh)
python3 -m venv .venv
source .venv/bin/activate

# instalar dependências
pip install -r requirements.txt
```

Executando a API (desenvolvimento)
```bash
uvicorn api:app --reload
```
A API é definida em `api.py`. Ajuste endpoints conforme necessário.

Treinamento do modelo
```bash
python main.py
```
Observações:
- Certifique-se de que a pasta `datasets/` contém os arquivos CSV necessários antes de executar.
- O treinamento pode demorar dependendo do tamanho dos dados e do hardware.

Estrutura esperada dos datasets
A pasta `datasets/` no repositório contém atualmente (exemplos demonstrados):
- `dataset.csv`
- `dataset2.csv`
- `symptom_Description.csv` (descrições de sintomas)
- `symptom_precaution.csv` (precauções por sintoma)
- `Symptom-severity.csv` (mapeamentos ou índices de severidade auxiliares)

Recomendações de formato:
-	Arquivos principais de amostras devem conter colunas como `Disease`, `Fever`, `Cough`, `Fatigue`, `Difficulty Breathing`, `Age`, `Gender`, `Blood Pressure`, `Cholesterol Level` (nem todas são obrigatórias, o pipeline ignora as ausentes).
-	Valores binários (Yes/No) são convertidos automaticamente para 1/0.

Saídas geradas
-	`anfis_results.png` — gráfico com loss, acurácia e matriz de confusão (salvo pelo script de plotagem).
-	(Se configurado no código) checkpoints / modelos salvos podem ser produzidos — ver `main.py` para salvar manualmente o modelo.

Reprodutibilidade
	O projeto inclui a função `set_seed(seed)` em `main.py` que fixa `numpy` e `tensorflow` para resultados reproduzíveis.
	Use sempre a mesma semente e a mesma versão das dependências para reproduzir resultados.

**Como funciona**
- Carregamento: o pipeline procura arquivos CSV na pasta `datasets/` e carrega automaticamente os arquivos compatíveis.
- Pré-processamento: normaliza e converte variáveis binárias (Yes/No), trata `Age`, `Gender`, `Blood Pressure` e `Cholesterol Level`, remove duplicatas e linhas com features críticas ausentes.
- Geração de labels: aplica regras heurísticas clínicas (função `create_severity_labels_improved`) para atribuir classes de severidade: 0=Autocuidado, 1=Consulta Médica, 2=Emergência.
- Treino: divide os dados em treino/validação/teste, normaliza features, calcula `class_weight` se necessário e treina um modelo aproximado de ANFIS (Keras) com callbacks (EarlyStopping, ReduceLROnPlateau).
- Avaliação: gera métricas (acurácia, relatório de classificação) e matriz de confusão; plota e salva gráficos em `anfis_results.png`.

**Como usar (passo a passo)**
1. Coloque os CSVs na pasta `datasets/`.
2. Crie e ative o ambiente virtual (ver seção Instalação).
3. Instale dependências: `pip install -r requirements.txt`.
4. Treine o modelo/execute o pipeline: `python main.py`.
   - O script `main.py` executa todo o fluxo: carregamento, pré-processamento, criação de labels, treino e avaliação.
5. Inicie a API (após treinar e salvar o modelo, se desejado): `uvicorn api:app --reload`.
6. Verifique as saídas: `anfis_results.png` e logs de treino. Se desejar exportar o modelo, adicione `model.save('model.h5')` em `main.py`.
# Sistema Neuro-Fuzzy para Triagem de Sintomas

**Integrantes**

- Diogo Buzatto (dbuzatto)
- Lucas Ferreira (lucasfeva)

Projeto que implementa um modelo neuro-fuzzy (ANFIS simplificado) para classificar gravidade de sintomas em 3 classes: Autocuidado, Consulta Medica e Emergencia. Inclui pipeline de dados, geracao de labels heuristicas, treinamento, avaliacao, artefatos salvos e API FastAPI pronta para consumo por frontend (React/Next).

## Visao rapida

- Pipeline unico em `main.py` (versao 3.0) com configuracao via `TrainingConfig` e seeds fixas.
- Limpeza + imputacao (mediana), mapeamento consistente de colunas binarias/categoricas e regra de severidade explicita.
- Oversampling opcional no treino para evitar colapso em classes raras (`use_oversampling=True` por padrao) + `class_weight`.
- Modelo ANFIS aproximado (fuzzificacao gaussiana + regras densas + consequente com L2/Dropout mais leves) com callbacks de EarlyStopping/ReduceLROnPlateau.
- Metricas: accuracy, macro-F1, micro-F1, matriz de confusao e relatorio de classificacao.
- Artefatos em `artifacts/`: modelo, scaler, imputer, label encoder, config (JSON) e grafico `anfis_results.png`.
- API em `api.py` usando `InferencePipeline` (`inference.py`) para previsoes consistentes.
- Opcional: validacao cruzada estratificada (k-fold) para avaliar variacao entre folds.

## Estrutura

- `main.py` - pipeline completo (dados -> preprocess -> labels -> split -> balanceamento -> treino -> avaliacao -> artefatos).
- `inference.py` - classe `InferencePipeline` para carregar artefatos e prever (aplica imputer + scaler antes da rede).
- `api.py` - FastAPI com `/health` e `/predict` usando `InferencePipeline`.
- `datasets/` - CSVs de treino (nao versionados aqui).
- `artifacts/` - saidas do treino (criado apos rodar `main.py`).
- `requirements.txt` - dependencias.

## Como rodar (treino)

```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py  # executa pipeline completo
```

Artefatos gerados em `artifacts/`: `anfis_symptom_triage_model.keras`, `scaler.pkl`, `imputer.pkl`, `label_encoder.pkl`, `config.json` e `anfis_results.png`.

### Configuracao

Edite a classe `TrainingConfig` em `main.py` para hiperparametros. Defaults relevantes para evitar colapso em classes raras:

- `num_mfs=2`, `num_rules=16`, `hidden1_units=48`, `hidden2_units=24`, `dropout1=0.2`, `dropout2=0.1`, `l2_reg=1e-3`.
- `use_oversampling=True` aplica oversampling no treino para balancear classes antes de treinar.
- `patience=25`, `epochs=150`, `learning_rate=1e-3`.
- `enable_cross_validation` pode ser ligado para StratifiedKFold (3 folds) e medir variacao.

### Validacao cruzada (opcional)

Habilite `enable_cross_validation = True` na `TrainingConfig` para rodar StratifiedKFold e ver macro-F1 medio por fold.

## Como rodar a API

Treine primeiro para gerar artefatos. Depois:

```bash
uvicorn api:app --reload
```

Requisicao de exemplo:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Fever": 1,
    "Cough": 1,
    "Fatigue": 0,
    "DifficultyBreathing": 1,
    "Age": 54,
    "Gender": 1,
    "BloodPressure": 2,
    "CholesterolLevel": 2
  }'
```

Resposta:

```json
{
  "predicted_class": "Emergencia",
  "probabilities": {
    "Autocuidado": 0.05,
    "Consulta Medica": 0.12,
    "Emergencia": 0.83
  }
}
```

### Payloads de exemplo (um por classe)

- Autocuidado (sintomas leves, sem fatores de risco):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Fever": 0,
    "Cough": 0,
    "Fatigue": 0,
    "DifficultyBreathing": 0,
    "Age": 25,
    "Gender": 0,
    "BloodPressure": 1,
    "CholesterolLevel": 1
  }'
```

- Consulta Medica (sintomas moderados + fator de risco):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Fever": 1,
    "Cough": 1,
    "Fatigue": 1,
    "DifficultyBreathing": 0,
    "Age": 52,
    "Gender": 1,
    "BloodPressure": 2,
    "CholesterolLevel": 2
  }'
```

- Emergencia (sintomas criticos):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Fever": 1,
    "Cough": 1,
    "Fatigue": 1,
    "DifficultyBreathing": 1,
    "Age": 70,
    "Gender": 1,
    "BloodPressure": 2,
    "CholesterolLevel": 2
  }'
```

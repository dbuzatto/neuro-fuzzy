import os
import glob
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import seaborn as sns

# Configuracao e reproducibilidade
@dataclass
class TrainingConfig:
    datasets_folder: str = "datasets"
    artifacts_dir: str = "artifacts"
    test_size: float = 0.20
    val_size: float = 0.20
    seed: int = 42
    num_mfs: int = 2
    num_rules: int = 16
    epochs: int = 150
    batch_size: int = 32
    learning_rate: float = 1e-3
    enable_cross_validation: bool = False
    num_folds: int = 3
    patience: int = 25
    min_delta: float = 1e-4
    hidden1_units: int = 48
    hidden2_units: int = 24
    dropout1: float = 0.2
    dropout2: float = 0.1
    l2_reg: float = 1e-3
    use_oversampling: bool = True


def set_seed(seed: int = 42) -> None:
    """Fixa seeds para reproducibilidade controlada."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Carregamento e preparacao dos dados
BINARY_COLUMNS = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
FEATURE_COLUMNS = [
    "Fever",
    "Cough",
    "Fatigue",
    "Difficulty Breathing",
    "Age",
    "Gender",
    "Blood Pressure",
    "Cholesterol Level",
]


def log_class_distribution(name: str, y: np.ndarray) -> None:
    """Mostra contagem e proporção das classes de severidade."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"[classes] {name}:")
    for u, c in zip(unique, counts):
        print(f"  classe {u}: {c} ({c/total:.2%})")


def load_all_datasets(datasets_folder: str) -> Dict[str, pd.DataFrame]:
    """Carrega todos os CSVs da pasta de datasets."""
    csv_files = sorted(glob.glob(os.path.join(datasets_folder, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {datasets_folder}")

    datasets = {}
    for path in csv_files:
        name = os.path.basename(path)
        datasets[name] = pd.read_csv(path)
        print(f"[dados] {name}: {datasets[name].shape[0]} linhas, {datasets[name].shape[1]} colunas")
    return datasets


def load_auxiliary_data(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Seleciona dataframes auxiliares se existirem."""
    aux = {}
    if "Symptom-severity.csv" in datasets:
        aux["severity"] = datasets["Symptom-severity.csv"]
    if "symptom_Description.csv" in datasets:
        aux["descriptions"] = datasets["symptom_Description.csv"]
    if "symptom_precaution.csv" in datasets:
        aux["precautions"] = datasets["symptom_precaution.csv"]
    return aux


def prepare_main_dataset(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combina datasets com o mesmo schema para formar o dataset principal."""
    expected_cols = {"Fever", "Cough", "Fatigue", "Difficulty Breathing", "Age", "Disease"}
    main_parts: List[pd.DataFrame] = []

    for name, df in datasets.items():
        if expected_cols.issubset(set(df.columns)):
            main_parts.append(df)
            print(f"[merge] usando {name} ({len(df)} registros)")

    if not main_parts:
        raise ValueError("Nenhum dataset com as colunas esperadas foi encontrado")

    combined = pd.concat(main_parts, ignore_index=True)
    print(f"[merge] dataset combinado: {combined.shape[0]} linhas")
    return combined


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder, SimpleImputer]:
    """Limpa, codifica e imputa faltantes."""
    data = df.copy()
    data = data.drop_duplicates().reset_index(drop=True)

    # Mapear colunas binarias Yes/No para 1/0
    for col in BINARY_COLUMNS:
        if col in data.columns:
            data[col] = data[col].map({"Yes": 1, "No": 0, 1: 1, 0: 0})

    if "Gender" in data.columns:
        data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0, 1: 1, 0: 0})

    if "Blood Pressure" in data.columns:
        bp_map = {"Low": 0, "Normal": 1, "High": 2, 0: 0, 1: 1, 2: 2}
        data["Blood Pressure"] = data["Blood Pressure"].map(bp_map)

    if "Cholesterol Level" in data.columns:
        chol_map = {"Low": 0, "Normal": 1, "High": 2, 0: 0, 1: 1, 2: 2}
        data["Cholesterol Level"] = data["Cholesterol Level"].map(chol_map)

    if "Age" in data.columns:
        data["Age"] = pd.to_numeric(data["Age"], errors="coerce")

    numeric_cols = [c for c in FEATURE_COLUMNS if c in data.columns]
    imputer = SimpleImputer(strategy="median")
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

    if "Disease" not in data.columns:
        raise ValueError("Coluna 'Disease' e obrigatoria para rotular severidade")

    label_encoder = LabelEncoder()
    data["Disease_Encoded"] = label_encoder.fit_transform(data["Disease"].astype(str))

    return data, label_encoder, imputer


def create_severity_labels_improved(data: pd.DataFrame, aux_data: Dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Gera labels de severidade usando regras heuristicas e fatores de risco."""
    emergency_diseases = {
        "Heart attack",
        "Stroke",
        "Pneumonia",
        "Sepsis",
        "AIDS",
        "Hepatitis",
        "Tuberculosis",
        "COVID-19",
        "Ebola Virus",
        "Myocardial Infarction",
        "Chronic Obstructive Pulmonary Disease",
        "Paralysis",
        "Brain Tumor",
        "Lung Cancer",
        "Liver Cancer",
        "Kidney Cancer",
        "Pancreatic Cancer",
    }

    consultation_diseases = {
        "Asthma",
        "Bronchitis",
        "Diabetes",
        "Hypertension",
        "Migraine",
        "Gastroenteritis",
        "Urinary Tract Infection",
        "Hyperthyroidism",
        "Hypothyroidism",
        "Depression",
        "Anxiety Disorders",
        "Arthritis",
    }

    def rule_row(row: pd.Series) -> int:
        disease = str(row.get("Disease", ""))
        age = float(row.get("Age", 0))
        fever = int(row.get("Fever", 0))
        cough = int(row.get("Cough", 0))
        fatigue = int(row.get("Fatigue", 0))
        diff_breath = int(row.get("Difficulty Breathing", 0))
        blood_pressure = row.get("Blood Pressure", np.nan)
        cholesterol = row.get("Cholesterol Level", np.nan)

        if any(tag in disease for tag in emergency_diseases):
            return 2
        if any(tag in disease for tag in consultation_diseases):
            return 1

        symptom_count = fever + cough + fatigue
        risk_factors = 0
        if not pd.isna(blood_pressure) and blood_pressure == 2:
            risk_factors += 1
        if not pd.isna(cholesterol) and cholesterol == 2:
            risk_factors += 1
        if age >= 50:
            risk_factors += 1

        if diff_breath and fever and age >= 60:
            return 2
        if diff_breath and fever and fatigue:
            return 2
        if age >= 70 and diff_breath:
            return 2
        if symptom_count >= 2 and risk_factors >= 1:
            return 1
        if fever and cough and fatigue:
            return 1
        if symptom_count <= 1 and risk_factors == 0:
            return 0
        return 1

    data["Severity"] = data.apply(rule_row, axis=1)
    return data


def normalize_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm, scaler


def oversample_minority(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Oversampling simples para balancear classes no conjunto de treino."""
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_balanced = []
    y_balanced = []
    for cls in unique:
        idx = np.where(y == cls)[0]
        need = max_count - len(idx)
        chosen = rng.choice(idx, size=need, replace=True) if need > 0 else np.array([], dtype=int)
        idx_all = np.concatenate([idx, chosen])
        X_balanced.append(X[idx_all])
        y_balanced.append(y[idx_all])

    X_out = np.vstack(X_balanced)
    y_out = np.concatenate(y_balanced)
    return X_out, y_out


# Modelo neuro-fuzzy (ANFIS simplificado)

class ANFISLayer(layers.Layer):
    """Camada de fuzzificacao com funcoes Gaussianas."""

    def __init__(self, num_mfs: int, **kwargs):
        super().__init__(**kwargs)
        self.num_mfs = num_mfs
        self.num_inputs = None

    def build(self, input_shape):
        self.num_inputs = int(input_shape[-1])
        self.mu = self.add_weight(
            name="mu",
            shape=(self.num_inputs, self.num_mfs),
            initializer="uniform",
            trainable=True,
        )
        self.sigma = self.add_weight(
            name="sigma",
            shape=(self.num_inputs, self.num_mfs),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        memberships = []
        for i in range(self.num_inputs):
            x = tf.expand_dims(inputs[:, i], axis=-1)
            mu = self.mu[i : i + 1, :]
            sigma = tf.abs(self.sigma[i : i + 1, :]) + 1e-6
            membership = tf.exp(-tf.square(x - mu) / (2.0 * tf.square(sigma)))
            memberships.append(membership)
        return tf.concat(memberships, axis=-1)


def build_anfis_model(input_dim: int, config: TrainingConfig, num_classes: int = 3) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="input")
    fuzz = ANFISLayer(config.num_mfs, name="fuzzification")(inputs)
    rules = layers.Dense(config.num_rules, activation="relu", name="rules")(fuzz)
    normalized = layers.BatchNormalization(name="normalization")(rules)
    consequent = layers.Dense(
        config.hidden1_units,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(config.l2_reg),
        name="consequent_1",
    )(normalized)
    consequent = layers.Dropout(config.dropout1)(consequent)
    consequent = layers.Dense(
        config.hidden2_units,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(config.l2_reg),
        name="consequent_2",
    )(consequent)
    consequent = layers.Dropout(config.dropout2)(consequent)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(consequent)
    model = keras.Model(inputs=inputs, outputs=outputs, name="ANFIS")
    return model


# Treinamento e avaliacao
def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: Dict[int, float],
    config: TrainingConfig,
):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        min_delta=config.min_delta,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )
    return history


def evaluate_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray):
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    micro_f1 = f1_score(y_test, y_pred, average="micro")

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Autocuidado", "Consulta Medica", "Emergencia"],
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print(f"[metricas] accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} micro_f1={micro_f1:.4f}")
    print(report)

    return y_pred, {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def plot_results(history: keras.callbacks.History, cm: np.ndarray, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history.history["loss"], label="treino", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="validacao", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="treino", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="validacao", linewidth=2)
    axes[1].set_title("Acuracia")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2])
    axes[2].set_title("Matriz de confusao")
    axes[2].set_xlabel("Predito")
    axes[2].set_ylabel("Real")
    axes[2].set_xticklabels(["Autocuidado", "Consulta", "Emergencia"], rotation=45)
    axes[2].set_yticklabels(["Autocuidado", "Consulta", "Emergencia"], rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[artefato] graficos salvos em {out_path}")


def cross_validate_model(X: np.ndarray, y: np.ndarray, config: TrainingConfig) -> List[float]:
    """Executa validacao cruzada estratificada opcional para medir variacao."""
    scores = []
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"[cv] fold {fold}/{config.num_folds}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_norm, X_val_norm, scaler = normalize_features(X_train, X_val)

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train,
        )
        class_weights = {cls: w for cls, w in zip(np.unique(y_train), class_weights)}

        model = build_anfis_model(X_train_norm.shape[1], config)
        history = train_model(
            model,
            X_train_norm,
            y_train,
            X_val_norm,
            y_val,
            class_weights,
            config,
        )
        _, metrics = evaluate_model(model, X_val_norm, y_val)
        scores.append(metrics["macro_f1"])
        print(f"[cv] macro_f1 fold {fold}: {metrics['macro_f1']:.4f}")
    print(f"[cv] macro_f1 medio: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    return scores


# Pipeline principal
def run_training(config: TrainingConfig):
    set_seed(config.seed)

    os.makedirs(config.artifacts_dir, exist_ok=True)

    print("=" * 70)
    print(" SISTEMA NEURO-FUZZY PARA TRIAGEM DE SINTOMAS ")
    print(" Versao 3.0 - pipeline organizado para API ")
    print("=" * 70)

    datasets = load_all_datasets(config.datasets_folder)
    aux_data = load_auxiliary_data(datasets)
    df = prepare_main_dataset(datasets)
    data, label_encoder, imputer = preprocess_data(df)
    data = create_severity_labels_improved(data, aux_data)

    available_features = [c for c in FEATURE_COLUMNS if c in data.columns]
    print(f"[features] usando: {available_features}")

    X = data[available_features].values
    y = data["Severity"].values

    log_class_distribution("dataset completo", y)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=config.val_size,
        random_state=config.seed,
        stratify=y_temp,
    )

    log_class_distribution("treino (antes do oversampling)", y_train)
    log_class_distribution("validacao", y_val)
    log_class_distribution("teste", y_test)

    if config.use_oversampling:
        X_train, y_train = oversample_minority(X_train, y_train, seed=config.seed)
        print("[balance] oversampling aplicado ao treino")
        log_class_distribution("treino (apos oversampling)", y_train)

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = {cls: w for cls, w in zip(np.unique(y_train), class_weights_array)}

    X_train_norm, X_test_norm, scaler = normalize_features(X_train, X_test)
    X_val_norm = scaler.transform(X_val)

    model = build_anfis_model(X_train_norm.shape[1], config)
    history = train_model(
        model,
        X_train_norm,
        y_train,
        X_val_norm,
        y_val,
        class_weights,
        config,
    )

    _, metrics = evaluate_model(model, X_test_norm, y_test)

    plot_path = os.path.join(config.artifacts_dir, "anfis_results.png")
    plot_results(history, metrics["confusion_matrix"], plot_path)

    model_path = os.path.join(config.artifacts_dir, "anfis_symptom_triage_model.keras")
    scaler_path = os.path.join(config.artifacts_dir, "scaler.pkl")
    encoder_path = os.path.join(config.artifacts_dir, "label_encoder.pkl")
    imputer_path = os.path.join(config.artifacts_dir, "imputer.pkl")
    config_path = os.path.join(config.artifacts_dir, "config.json")

    model.save(model_path)
    import joblib

    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(imputer, imputer_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"[artefato] modelo salvo em {model_path}")
    print(f"[artefato] scaler salvo em {scaler_path}")
    print(f"[artefato] label encoder salvo em {encoder_path}")
    print(f"[artefato] imputer salvo em {imputer_path}")
    print(f"[artefato] config salva em {config_path}")

    if config.enable_cross_validation:
        cross_validate_model(X, y, config)

    # Exemplo de predicao rapida
    example = np.array([[1, 1, 1, 1, 65, 1, 2, 2]])
    example_norm = scaler.transform(example)
    pred = model.predict(example_norm, verbose=0)
    predicted_class = int(np.argmax(pred))
    classes = ["Autocuidado", "Consulta Medica", "Emergencia"]
    print(f"[exemplo] classe prevista: {classes[predicted_class]}")

    return model, scaler, history


def main():
    config = TrainingConfig()
    run_training(config)


if __name__ == "__main__":
    main()

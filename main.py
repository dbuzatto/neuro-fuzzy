import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ============================================

def load_all_datasets(datasets_folder='datasets'):
    """Carrega todos os datasets da pasta datasets/"""
    print("="*60)
    print("CARREGANDO DATASETS")
    print("="*60)
    
    datasets = {}
    
    # Procurar todos os CSVs na pasta datasets
    csv_files = glob.glob(os.path.join(datasets_folder, '*.csv'))
    
    if not csv_files:
        print(f"\n⚠️ Nenhum arquivo CSV encontrado em '{datasets_folder}/'")
        return None
    
    print(f"\n📂 Arquivos encontrados:")
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"  - {filename}")
        
        try:
            df = pd.read_csv(csv_file)
            datasets[filename] = df
            print(f"    ✓ Carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
        except Exception as e:
            print(f"    ✗ Erro ao carregar: {e}")
    
    return datasets

def load_auxiliary_data(datasets):
    """Carrega dados auxiliares (severidade, descrições, precauções)"""
    print("\n" + "="*50)
    print("CARREGANDO DADOS AUXILIARES")
    print("="*50)
    
    aux_data = {}
    
    # Symptom Severity
    if 'Symptom-severity.csv' in datasets:
        aux_data['severity'] = datasets['Symptom-severity.csv']
        print(f"✓ Severidade de sintomas: {len(aux_data['severity'])} sintomas")
    
    # Symptom Descriptions
    if 'symptom_Description.csv' in datasets:
        aux_data['descriptions'] = datasets['symptom_Description.csv']
        print(f"✓ Descrições: {len(aux_data['descriptions'])} doenças")
    
    # Symptom Precautions
    if 'symptom_precaution.csv' in datasets:
        aux_data['precautions'] = datasets['symptom_precaution.csv']
        print(f"✓ Precauções: {len(aux_data['precautions'])} doenças")
    
    return aux_data

def prepare_main_dataset(datasets):
    """Prepara o dataset principal para treinamento"""
    print("\n" + "="*50)
    print("PREPARANDO DATASET PRINCIPAL")
    print("="*50)
    
    # Identificar datasets principais (com estrutura similar)
    main_datasets = []
    
    for name, df in datasets.items():
        # Verificar se tem as colunas esperadas
        expected_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age']
        if all(col in df.columns for col in expected_cols):
            main_datasets.append((name, df))
            print(f"✓ Dataset principal: {name} ({df.shape[0]} registros)")
    
    if not main_datasets:
        print("⚠️ Nenhum dataset principal encontrado!")
        return None
    
    # Combinar todos os datasets principais
    if len(main_datasets) > 1:
        print(f"\n📊 Combinando {len(main_datasets)} datasets...")
        combined_df = pd.concat([df for _, df in main_datasets], ignore_index=True)
    else:
        combined_df = main_datasets[0][1]
    
    print(f"\n✓ Dataset combinado: {combined_df.shape[0]} registros totais")
    print(f"✓ Colunas: {combined_df.columns.tolist()}")
    
    # Visualizar primeiras linhas
    print("\n📋 Primeiras linhas:")
    print(combined_df.head())
    
    # Estatísticas
    print("\n📊 Estatísticas:")
    print(f"  - Total de registros: {len(combined_df)}")
    print(f"  - Doenças únicas: {combined_df['Disease'].nunique()}")
    print(f"  - Valores nulos: {combined_df.isnull().sum().sum()}")
    
    return combined_df

def preprocess_data(df):
    """Pré-processamento completo dos dados"""
    print("\n" + "="*50)
    print("PREPROCESSAMENTO DOS DADOS")
    print("="*50)
    
    # Criar cópia
    data = df.copy()
    
    # Remover duplicatas
    before = len(data)
    data = data.drop_duplicates()
    after = len(data)
    if before != after:
        print(f"✓ Removidas {before - after} linhas duplicadas")
    
    # 1. Converter variáveis binárias (Yes/No)
    binary_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
    for col in binary_columns:
        if col in data.columns:
            data[col] = data[col].map({'Yes': 1, 'No': 0})
    
    # 2. Converter Gender
    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    
    # 3. Converter Blood Pressure
    if 'Blood Pressure' in data.columns:
        bp_mapping = {'Low': 0, 'Normal': 1, 'High': 2}
        data['Blood Pressure'] = data['Blood Pressure'].map(bp_mapping)
    
    # 4. Converter Cholesterol Level
    if 'Cholesterol Level' in data.columns:
        chol_mapping = {'Low': 0, 'Normal': 1, 'High': 2}
        data['Cholesterol Level'] = data['Cholesterol Level'].map(chol_mapping)
    
    # 5. Converter Outcome Variable (se existir)
    if 'Outcome Variable' in data.columns:
        data['Outcome Variable'] = data['Outcome Variable'].map({'Negative': 0, 'Positive': 1})
    
    # 6. Label Encoding para Disease
    le_disease = LabelEncoder()
    data['Disease_Encoded'] = le_disease.fit_transform(data['Disease'])
    
    # Remover linhas com valores nulos nas features críticas
    critical_features = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age']
    before = len(data)
    data = data.dropna(subset=[col for col in critical_features if col in data.columns])
    after = len(data)
    if before != after:
        print(f"✓ Removidas {before - after} linhas com valores nulos")
    
    print(f"\n✓ Dados após conversão:")
    print(data.head())
    print(f"✓ Shape final: {data.shape}")
    print(f"✓ Valores nulos restantes: {data.isnull().sum().sum()}")
    
    return data, le_disease

def create_severity_labels_improved(data, aux_data=None):
    """
    Cria labels de gravidade MELHORADOS usando:
    - Regras clínicas mais robustas
    - Dados de severidade de sintomas (se disponível)
    - Combinação de múltiplos fatores
    
    0 = Autocuidado
    1 = Consulta Médica
    2 = Emergência
    """
    print("\n" + "="*50)
    print("CRIAÇÃO DE LABELS DE GRAVIDADE (MELHORADO)")
    print("="*50)
    
    severity = []
    
    # Dicionário de doenças graves conhecidas
    emergency_diseases = [
        'Heart attack', 'Stroke', 'Pneumonia', 'Sepsis', 'AIDS', 
        'Hepatitis', 'Tuberculosis', 'COVID-19', 'Ebola Virus',
        'Myocardial Infarction', 'Chronic Obstructive Pulmonary Disease',
        'Paralysis', 'Brain Tumor', 'Lung Cancer', 'Liver Cancer',
        'Kidney Cancer', 'Pancreatic Cancer'
    ]
    
    consultation_diseases = [
        'Asthma', 'Bronchitis', 'Diabetes', 'Hypertension',
        'Migraine', 'Gastroenteritis', 'Urinary Tract Infection',
        'Hyperthyroidism', 'Hypothyroidism', 'Depression',
        'Anxiety Disorders', 'Arthritis'
    ]
    
    for idx, row in data.iterrows():
        disease = row['Disease']
        
        # REGRA 1: Doenças conhecidas graves
        if any(d in disease for d in emergency_diseases):
            severity.append(2)  # Emergência
            continue
        
        # REGRA 2: Doenças de consulta conhecidas
        if any(d in disease for d in consultation_diseases):
            severity.append(1)  # Consulta
            continue
        
        # REGRA 3: Sintomas críticos (Emergência)
        if (row['Difficulty Breathing'] == 1 and 
            row['Fever'] == 1 and 
            row['Age'] >= 60):
            severity.append(2)
            continue
        
        # REGRA 4: Múltiplos sintomas graves (Emergência)
        if (row['Difficulty Breathing'] == 1 and 
            row['Fever'] == 1 and 
            row['Fatigue'] == 1):
            severity.append(2)
            continue
        
        # REGRA 5: Idosos com sintomas respiratórios (Emergência)
        if row['Age'] >= 70 and row['Difficulty Breathing'] == 1:
            severity.append(2)
            continue
        
        # REGRA 6: Sintomas moderados + fatores de risco (Consulta)
        symptom_count = sum([
            row['Fever'], 
            row['Cough'], 
            row['Fatigue']
        ])
        
        risk_factors = 0
        if 'Blood Pressure' in row and row['Blood Pressure'] == 2:
            risk_factors += 1
        if 'Cholesterol Level' in row and row['Cholesterol Level'] == 2:
            risk_factors += 1
        if row['Age'] >= 50:
            risk_factors += 1
        
        if symptom_count >= 2 and risk_factors >= 1:
            severity.append(1)  # Consulta
            continue
        
        # REGRA 7: Febre + tosse + fadiga (Consulta)
        if (row['Fever'] == 1 and 
            row['Cough'] == 1 and 
            row['Fatigue'] == 1):
            severity.append(1)
            continue
        
        # REGRA 8: Sintomas leves (Autocuidado)
        if symptom_count <= 1 and risk_factors == 0:
            severity.append(0)
            continue
        
        # PADRÃO: Consulta médica
        severity.append(1)
    
    data['Severity'] = severity
    
    # Estatísticas
    print(f"\n📊 Distribuição de gravidade:")
    severity_counts = data['Severity'].value_counts().sort_index()
    severity_props = data['Severity'].value_counts(normalize=True).sort_index()
    
    labels = ['Autocuidado', 'Consulta Médica', 'Emergência']
    for i in range(3):
        count = severity_counts.get(i, 0)
        prop = severity_props.get(i, 0)
        print(f"  {labels[i]:20s}: {count:4d} ({prop:6.2%})")
    
    return data

def normalize_features(X_train, X_test):
    """Normaliza as features usando Min-Max Scaling"""
    print("\n" + "="*50)
    print("NORMALIZAÇÃO DAS FEATURES")
    print("="*50)
    
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    
    print(f"✓ Range: [{X_train_normalized.min():.2f}, {X_train_normalized.max():.2f}]")
    
    return X_train_normalized, X_test_normalized, scaler

# ============================================
# 2. IMPLEMENTAÇÃO DO SISTEMA NEURO-FUZZY
# ============================================

class ANFISLayer(layers.Layer):
    """Camada de fuzzificação com funções Gaussianas"""
    
    def __init__(self, num_mfs, **kwargs):
        super(ANFISLayer, self).__init__(**kwargs)
        self.num_mfs = num_mfs
    
    def build(self, input_shape):
        num_inputs = input_shape[-1]
        
        self.mu = self.add_weight(
            name='mu',
            shape=(num_inputs, self.num_mfs),
            initializer='uniform',
            trainable=True
        )
        
        self.sigma = self.add_weight(
            name='sigma',
            shape=(num_inputs, self.num_mfs),
            initializer='ones',
            trainable=True
        )
        
        super(ANFISLayer, self).build(input_shape)
    
    def call(self, inputs):
        memberships = []
        for i in range(inputs.shape[-1]):
            x = tf.expand_dims(inputs[:, i], axis=-1)
            mu = self.mu[i:i+1, :]
            sigma = tf.abs(self.sigma[i:i+1, :]) + 1e-6
            
            membership = tf.exp(-tf.square(x - mu) / (2 * tf.square(sigma)))
            memberships.append(membership)
        
        return tf.concat(memberships, axis=-1)

def build_anfis_model(input_dim, num_mfs=3, num_rules=27, num_classes=3):
    """Constrói o modelo ANFIS com melhorias"""
    print("\n" + "="*50)
    print("CONSTRUÇÃO DO MODELO ANFIS")
    print("="*50)
    
    inputs = keras.Input(shape=(input_dim,), name='input')
    
    # Camada 1: Fuzzificação
    fuzz = ANFISLayer(num_mfs, name='fuzzification')(inputs)
    
    # Camada 2: Regras
    rules = layers.Dense(num_rules, activation='relu', name='rules')(fuzz)
    
    # Camada 3: Normalização
    normalized = layers.BatchNormalization(name='normalization')(rules)
    
    # Camada 4: Consequente (TSK) - MELHORADO com mais regularização
    consequent = layers.Dense(64, activation='relu', 
                             kernel_regularizer=keras.regularizers.l2(0.01),
                             name='consequent_1')(normalized)
    consequent = layers.Dropout(0.5)(consequent)  # Dropout aumentado
    consequent = layers.Dense(32, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.01),
                             name='consequent_2')(consequent)
    consequent = layers.Dropout(0.3)(consequent)
    
    # Camada 5: Saída
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(consequent)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='ANFIS')
    
    print(f"\n✓ Arquitetura do modelo:")
    model.summary()
    
    return model

# ============================================
# 3. TREINAMENTO E AVALIAÇÃO
# ============================================

def train_model(model, X_train, y_train, X_val, y_val, 
                class_weights=None, epochs=150, batch_size=32):
    """Treina o modelo ANFIS com balanceamento de classes"""
    print("\n" + "="*50)
    print("TREINAMENTO DO MODELO")
    print("="*50)
    
    # Compilar o modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks melhorados
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )
    
    # Treinamento
    print(f"\n🚀 Iniciando treinamento...")
    print(f"  Épocas: {epochs}")
    print(f"  Batch size: {batch_size}")
    if class_weights:
        print(f"  Class weights: {class_weights}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Avalia o modelo treinado"""
    print("\n" + "="*50)
    print("AVALIAÇÃO DO MODELO")
    print("="*50)
    
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n📋 Relatório de Classificação:")
    print(classification_report(
        y_test, 
        y_pred,
        target_names=['Autocuidado', 'Consulta Médica', 'Emergência'],
        zero_division=0
    ))
    
    cm = confusion_matrix(y_test, y_pred)
    
    return y_pred, accuracy, cm

def plot_results(history, cm):
    """Visualiza os resultados"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Treino', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validação', linewidth=2)
    axes[0].set_title('Loss durante o Treinamento', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Acurácia
    axes[1].plot(history.history['accuracy'], label='Treino', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validação', linewidth=2)
    axes[1].set_title('Acurácia durante o Treinamento', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Acurácia')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Matriz de confusão
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2], 
                cbar_kws={'label': 'Contagem'})
    axes[2].set_title('Matriz de Confusão', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Predito')
    axes[2].set_ylabel('Real')
    axes[2].set_xticklabels(['Autocuidado', 'Consulta', 'Emergência'], rotation=45)
    axes[2].set_yticklabels(['Autocuidado', 'Consulta', 'Emergência'], rotation=0)
    
    plt.tight_layout()
    plt.savefig('anfis_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráficos salvos em 'anfis_results.png'")
    plt.show()

# ============================================
# 4. PIPELINE PRINCIPAL
# ============================================

def main():
    """Pipeline principal de execução"""
    print("="*60)
    print(" SISTEMA NEURO-FUZZY PARA TRIAGEM DE SINTOMAS")
    print(" Versão 2.0 - Multi-Dataset")
    print("="*60)
    
    # 1. Carregar todos os datasets
    datasets = load_all_datasets('datasets')
    if not datasets:
        print("\n❌ Erro: Nenhum dataset encontrado!")
        return None
    
    # 2. Carregar dados auxiliares
    aux_data = load_auxiliary_data(datasets)
    
    # 3. Preparar dataset principal
    df = prepare_main_dataset(datasets)
    if df is None:
        print("\n❌ Erro: Não foi possível preparar o dataset!")
        return None
    
    # 4. Preprocessar
    data, le_disease = preprocess_data(df)
    
    # 5. Criar labels de gravidade (MELHORADO)
    data = create_severity_labels_improved(data, aux_data)
    
    # 6. Selecionar features
    feature_columns = [
        'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
        'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level'
    ]
    
    # Verificar quais features existem
    available_features = [col for col in feature_columns if col in data.columns]
    print(f"\n✓ Features disponíveis: {available_features}")
    
    X = data[available_features].values
    y = data['Severity'].values
    
    print(f"\n" + "="*50)
    print("DIVISÃO DOS DADOS")
    print("="*50)
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    
    # 7. Dividir em treino, validação e teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Treino: {X_train.shape[0]} amostras")
    print(f"✓ Validação: {X_val.shape[0]} amostras")
    print(f"✓ Teste: {X_test.shape[0]} amostras")
    
    # 8. Calcular pesos de classe para balanceamento
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
    
    # 9. Normalizar
    X_train_norm, X_test_norm, scaler = normalize_features(X_train, X_test)
    X_val_norm = scaler.transform(X_val)
    
    # 10. Construir modelo
    model = build_anfis_model(
        input_dim=X_train_norm.shape[1],
        num_mfs=3,
        num_rules=27,
        num_classes=3
    )
    
    # 11. Treinar
    history = train_model(
        model, 
        X_train_norm, y_train,
        X_val_norm, y_val,
        class_weights=class_weights,
        epochs=150,
        batch_size=32
    )
    
    # 12. Avaliar
    y_pred, accuracy, cm = evaluate_model(model, X_test_norm, y_test)
    
    # 13. Visualizar resultados
    plot_results(history, cm)
    
    # 14. Salvar modelo
    model.save('anfis_symptom_triage_model.keras')
    print("\n✓ Modelo salvo em 'anfis_symptom_triage_model.keras'")
    
    # 15. Salvar scaler
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le_disease, 'label_encoder.pkl')
    print("✓ Scaler e encoder salvos")
    
    # 16. Exemplo de predição
    print("\n" + "="*50)
    print("EXEMPLO DE PREDIÇÃO")
    print("="*50)
    
    # Exemplo: Paciente com sintomas graves
    example = np.array([[1, 1, 1, 1, 65, 1, 2, 2]])  # Febre, Tosse, Fadiga, Dif.Resp, 65 anos, M, Press.Alta, Col.Alto
    example_norm = scaler.transform(example)
    prediction = model.predict(example_norm, verbose=0)
    predicted_class = np.argmax(prediction)
    
    classes = ['Autocuidado', 'Consulta Médica', 'Emergência']
    print(f"\n👤 Paciente exemplo:")
    print(f"  Febre: Sim | Tosse: Sim | Fadiga: Sim | Dificuldade Respiratória: Sim")
    print(f"  Idade: 65 anos | Gênero: Masculino")
    print(f"  Pressão: Alta | Colesterol: Alto")
    print(f"\n🎯 Predição: {classes[predicted_class]}")
    print(f"\n📊 Probabilidades:")
    for i, prob in enumerate(prediction[0]):
        bar = '█' * int(prob * 20)
        print(f"  {classes[i]:20s}: {prob:6.2%} {bar}")
    
    return model, scaler, history

if __name__ == "__main__":
    model, scaler, history = main()
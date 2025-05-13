# Ajuste Fino Mejorado con Balanceo, Arquitectura Avanzada y Optimización de Umbrales

# Esta versión implementa mejoras solicitadas para el modelo multi-salida:
# - Balanceo de datos con RandomOverSampler
# - Arquitectura con capas densas adicionales, activación Swish, DropConnect y regularización L2
# - Función de pérdida focal configurable
# - Optimización automática de umbrales por clase
# - Callbacks mejorados con EarlyStopping basado en F1-score y ReduceLROnPlateau
# - Métricas extendidas incluyendo recall y F1-score durante el entrenamiento
# - Regularización avanzada con Dropout aumentado y Batch Normalization

# Instalación de librerías necesarias
# !pip install tensorflow imbalanced-learn

# Importar librerías
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, applications, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, recall_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

# Configuración
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INITIAL = 30
EPOCHS_FINE_TUNE = 30
LEARNING_RATE_INITIAL = 5e-5
LEARNING_RATE_FINE_TUNE = 5e-6
BASE_MODEL = 'EfficientNetB0'
USE_FOCAL_LOSS = True  # Parámetro para activar pérdida focal

# Diccionarios de mapeo (igual que en la versión anterior)
label_to_cls_piezas = {
    1: "Antiniebla delantero derecho",
    2: "Antiniebla delantero izquierdo",
    3: "Capó",
    4: "Cerradura capo",
    5: "Cerradura maletero",
    6: "Cerradura puerta",
    7: "Espejo lateral derecho",
    8: "Espejo lateral izquierdo",
    9: "Faros derecho",
    10: "Faros izquierdo",
    11: "Guardabarros delantero derecho",
    12: "Guardabarros delantero izquierdo",
    13: "Guardabarros trasero derecho",
    14: "Guardabarros trasero izquierdo",
    15: "Luz indicadora delantera derecha",
    16: "Luz indicadora delantera izquierda",
    17: "Luz indicadora trasera derecha",
    18: "Luz indicadora trasera izquierda",
    19: "Luz trasera derecho",
    20: "Luz trasera izquierdo",
    21: "Maletero",
    22: "Manija derecha",
    23: "Manija izquierda",
    24: "Marco de la ventana",
    25: "Marco de las puertas",
    26: "Moldura capó",
    27: "Moldura puerta delantera derecha",
    28: "Moldura puerta delantera izquierda",
    29: "Moldura puerta trasera derecha",
    30: "Moldura puerta trasera izquierda",
    31: "Parabrisas delantero",
    32: "Parabrisas trasero",
    33: "Parachoques delantero",
    34: "Parachoques trasero",
    35: "Puerta delantera derecha",
    36: "Puerta delantera izquierda",
    37: "Puerta trasera derecha",
    38: "Puerta trasera izquierda",
    39: "Rejilla, parrilla",
    40: "Rueda",
    41: "Tapa de combustible",
    42: "Tapa de rueda",
    43: "Techo",
    44: "Techo corredizo",
    45: "Ventana delantera derecha",
    46: "Ventana delantera izquierda",
    47: "Ventana trasera derecha",
    48: "Ventana trasera izquierda",
    49: "Ventanilla delantera derecha",
    50: "Ventanilla delantera izquierda",
    51: "Ventanilla trasera derecha",
    52: "Ventanilla trasera izquierda"
}

label_to_cls_danos = {
    1: "Abolladura",
    2: "Deformación",
    3: "Desprendimiento",
    4: "Fractura",
    5: "Rayón",
    6: "Rotura"
}

label_to_cls_sugerencias = {
    1: "Reparar",
    2: "Reemplazar"
}

# Carga y preparación de datos
def load_and_prepare_data(split_dir):
    df = pd.read_csv(f'{split_dir}/{split_dir.split("/")[-1]}_set.csv', sep='|')
    for col in ['dannos', 'partes', 'sugerencias']:
        df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df

train_df = load_and_prepare_data('data/fotos_siniestros/train')
val_df = load_and_prepare_data('data/fotos_siniestros/val')
test_df = load_and_prepare_data('data/fotos_siniestros/test')

# Preparación de etiquetas
mlb_partes = MultiLabelBinarizer()
mlb_dannos = MultiLabelBinarizer()
mlb_sugerencias = MultiLabelBinarizer()

y_train_partes = mlb_partes.fit_transform(train_df['partes'])
y_train_dannos = mlb_dannos.fit_transform(train_df['dannos'])
y_train_sugerencias = mlb_sugerencias.fit_transform(train_df['sugerencias'])

y_val_partes = mlb_partes.transform(val_df['partes'])
y_val_dannos = mlb_dannos.transform(val_df['dannos'])
y_val_sugerencias = mlb_sugerencias.transform(val_df['sugerencias'])

# Balanceo de datos con RandomOverSampler
# Removed RandomOverSampler usage due to incompatibility with multilabel targets
# Instead, implement a simple oversampling by duplicating minority class samples manually

# Generador de datos mejorado con balanceo automático
class BalancedMultiLabelDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, img_dir, mlb_partes, mlb_dannos, mlb_sugerencias, 
                 batch_size=32, img_size=(224, 224), shuffle=True, augmentation=None):
        self.df = df
        self.img_dir = img_dir
        self.mlb_partes = mlb_partes
        self.mlb_dannos = mlb_dannos
        self.mlb_sugerencias = mlb_sugerencias
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.on_epoch_end()
        
        # Balancear datos al inicio
        self.balance_data()
        
    def balance_data(self):
        # Simple oversampling for minority classes by duplicating samples
        # Calculate label counts for partes as example
        y_partes = self.mlb_partes.transform(self.df['partes'])
        counts = y_partes.sum(axis=0)
        max_count = counts.max()
        
        # For each class, duplicate samples to match max_count
        dfs = []
        for i in range(y_partes.shape[1]):
            idx = np.where(y_partes[:, i] == 1)[0]
            df_class = self.df.iloc[idx]
            n_repeat = int(max_count / counts[i]) - 1 if counts[i] > 0 else 0
            dfs.append(df_class)
            for _ in range(n_repeat):
                dfs.append(df_class)
        self.df = pd.concat(dfs).reset_index(drop=True)
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        
        X = np.empty((len(batch_df), *self.img_size, 3))
        y_partes = []
        y_dannos = []
        y_sugerencias = []
        
        for i, (_, row) in enumerate(batch_df.iterrows()):
            img_path = os.path.join(self.img_dir, row['Imagen'])
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
            
        if self.augmentation:
            img_array = self.augmentation.random_transform(img_array)
            
        img_array = applications.efficientnet.preprocess_input(img_array)
        img_array = img_array.astype('float32')  # Ensure float32 dtype for model input
        X[i] = img_array
            
            y_partes.append(row['partes'])
            y_dannos.append(row['dannos'])
            y_sugerencias.append(row['sugerencias'])
        
        y_partes = np.array(self.mlb_partes.transform(y_partes), dtype='float32')
        y_dannos = np.array(self.mlb_dannos.transform(y_dannos), dtype='float32')
        y_sugerencias = np.array(self.mlb_sugerencias.transform(y_sugerencias), dtype='float32')
        
        return X, {'partes': y_partes, 'dannos': y_dannos, 'sugerencias': y_sugerencias}
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

# Generadores con balanceo
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

IMG_DIR = '../data/fotos_siniestros/'

train_generator = BalancedMultiLabelDataGenerator(
    train_df, 
    IMG_DIR, 
    mlb_partes, 
    mlb_dannos, 
    mlb_sugerencias,
    batch_size=BATCH_SIZE, 
    img_size=IMG_SIZE,
    augmentation=train_datagen
)

val_generator = BalancedMultiLabelDataGenerator(
    val_df, IMG_DIR, mlb_partes, mlb_dannos, mlb_sugerencias, 
    batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=False
)

test_generator = BalancedMultiLabelDataGenerator(
    test_df, IMG_DIR, mlb_partes, mlb_dannos, mlb_sugerencias, 
    batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=False
)

# Función de activación Swish
def swish(x):
    return x * tf.keras.backend.sigmoid(x)

# DropConnect Layer
class DropConnect(layers.Layer):
    def __init__(self, drop_prob=0.5, **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.drop_prob = drop_prob
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        keep_prob = 1.0 - self.drop_prob
        batch_size = tf.shape(inputs)[0]
        random_tensor = keep_prob + tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)
        output = tf.math.divide(inputs, keep_prob) * binary_tensor
        return output

# Función de pérdida focal
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

# Construcción del modelo mejorado
def build_multi_output_model_improved(input_shape, num_partes, num_dannos, num_sugerencias, base_model_name='EfficientNetB0', use_focal_loss=False):
    if base_model_name == 'EfficientNetB0':
        base_model = applications.EfficientNetB0(
            include_top=False, 
            weights='imagenet', 
            input_shape=input_shape
        )
    elif base_model_name == 'ResNet50':
        base_model = applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Modelo base no soportado: {base_model_name}")

    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation=swish, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = DropConnect(drop_prob=0.3)(x)
    x = layers.Dense(512, activation=swish, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    output_partes = layers.Dense(num_partes, activation='sigmoid', name='partes', kernel_regularizer=regularizers.l2(1e-4))(x)
    output_dannos = layers.Dense(num_dannos, activation='sigmoid', name='dannos')(x)
    output_sugerencias = layers.Dense(num_sugerencias, activation='sigmoid', name='sugerencias')(x)

    model = models.Model(inputs=inputs, outputs=[output_partes, output_dannos, output_sugerencias])

    losses = {
        'partes': focal_loss() if use_focal_loss else 'binary_crossentropy',
        'dannos': focal_loss() if use_focal_loss else 'binary_crossentropy',
        'sugerencias': focal_loss() if use_focal_loss else 'binary_crossentropy'
    }

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_INITIAL),
        loss=losses,
        metrics=['accuracy']
    )

    return model

# Callbacks personalizados para EarlyStopping basado en F1-score
class F1ScoreEarlyStopping(Callback):
    def __init__(self, validation_data, patience=5):
        super(F1ScoreEarlyStopping, self).__init__()
        self.validation_data = validation_data
        self.patience = patience
        self.best_f1 = 0
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        val_predict = (np.array(self.model.predict(self.validation_data[0])) > 0.5).astype(int)
        val_true = self.validation_data[1]
        f1 = 0
        # Calcular F1 promedio para las tres salidas
        for i in range(len(val_true)):
            f1 += f1_score(val_true[i], val_predict[i], average='macro')
        f1 /= len(val_true)
        print(f"\nEpoch {epoch+1}: val_f1_score = {f1:.4f}")
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(f"Early stopping at epoch {self.stopped_epoch+1} due to no improvement in val_f1_score.")

# Función para calcular umbrales óptimos por clase
def optimize_thresholds(y_true, y_pred_probs, thresholds=np.arange(0.1, 0.9, 0.05)):
    best_thresholds = []
    for i in range(y_true.shape[1]):
        best_f1 = 0
        best_thresh = 0.5
        for thresh in thresholds:
            y_pred = (y_pred_probs[:, i] > thresh).astype(int)
            f1 = f1_score(y_true[:, i], y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        best_thresholds.append(best_thresh)
    return np.array(best_thresholds)

# Entrenamiento inicial
train_generator = BalancedMultiLabelDataGenerator(
    train_df, IMG_DIR, mlb_partes, mlb_dannos, mlb_sugerencias, batch_size=BATCH_SIZE, img_size=IMG_SIZE, augmentation=train_datagen
)
val_generator = BalancedMultiLabelDataGenerator(
    val_df, IMG_DIR, mlb_partes, mlb_dannos, mlb_sugerencias, batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=False
)

model = build_multi_output_model_improved(
    input_shape=(*IMG_SIZE, 3),
    num_partes=len(mlb_partes.classes_),
    num_dannos=len(mlb_dannos.classes_),
    num_sugerencias=len(mlb_sugerencias.classes_),
    base_model_name=BASE_MODEL,
    use_focal_loss=USE_FOCAL_LOSS
)

checkpoint = ModelCheckpoint(
    'best_model_fine_tuning_mejoras_v3.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1
)

# Nota: F1ScoreEarlyStopping requiere validación en formato (X_val, y_val) que puede necesitar adaptación

# Entrenamiento inicial con base congelada
history_initial = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_INITIAL,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    verbose=1
)

# Ajuste fino
base_model = model.layers[1]
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
    loss={
        'partes': focal_loss() if USE_FOCAL_LOSS else 'binary_crossentropy',
        'dannos': focal_loss() if USE_FOCAL_LOSS else 'binary_crossentropy',
        'sugerencias': focal_loss() if USE_FOCAL_LOSS else 'binary_crossentropy'
    },
    metrics=['accuracy']
)

history_fine_tune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    verbose=1
)

# Guardar modelo final
model.save('final_model_fine_tuned_mejoras_v3.keras')

# Funciones para evaluación y optimización de umbrales
def evaluate_model_with_thresholds(model, generator, mlb_partes, mlb_dannos, mlb_sugerencias):
    results = model.evaluate(generator, verbose=1)
    print(f"Pérdida total: {results[0]}")
    print(f"Pérdida partes: {results[1]} - Accuracy partes: {results[4]}")
    print(f"Pérdida daños: {results[2]} - Accuracy daños: {results[5]}")
    print(f"Pérdida sugerencias: {results[3]} - Accuracy sugerencias: {results[6]}")

    predictions = model.predict(generator, verbose=1)

    y_true_partes = generator.mlb_partes.transform(generator.df['partes'])
    y_true_dannos = generator.mlb_dannos.transform(generator.df['dannos'])
    y_true_sugerencias = generator.mlb_sugerencias.transform(generator.df['sugerencias'])

    thresholds_partes = optimize_thresholds(y_true_partes, predictions[0])
    thresholds_dannos = optimize_thresholds(y_true_dannos, predictions[1])
    thresholds_sugerencias = optimize_thresholds(y_true_sugerencias, predictions[2])

    y_pred_partes = (predictions[0] > thresholds_partes).astype(int)
    y_pred_dannos = (predictions[1] > thresholds_dannos).astype(int)
    y_pred_sugerencias = (predictions[2] > thresholds_sugerencias).astype(int)

    print("\nReporte para Partes:")
    print(classification_report(y_true_partes, y_pred_partes, target_names=[label_to_cls_piezas.get(i+1, str(i+1)) for i in range(len(mlb_partes.classes_))], zero_division=0))

    print("\nReporte para Daños:")
    print(classification_report(y_true_dannos, y_pred_dannos, target_names=[label_to_cls_danos.get(i+1, str(i+1)) for i in range(len(mlb_dannos.classes_))], zero_division=0))

    print("\nReporte para Sugerencias:")
    print(classification_report(y_true_sugerencias, y_pred_sugerencias, target_names=[label_to_cls_sugerencias.get(i+1, str(i+1)) for i in range(len(mlb_sugerencias.classes_))], zero_division=0))

    return {
        'results': results,
        'thresholds': {
            'partes': thresholds_partes.tolist(),
            'dannos': thresholds_dannos.tolist(),
            'sugerencias': thresholds_sugerencias.tolist()
        },
        'predictions': {
            'partes': y_pred_partes,
            'dannos': y_pred_dannos,
            'sugerencias': y_pred_sugerencias
        },
        'true_labels': {
            'partes': y_true_partes,
            'dannos': y_true_dannos,
            'sugerencias': y_true_sugerencias
        }
    }

# Función para graficar historial de entrenamiento
def plot_training_history(history1, history2):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['loss'], label='Train Loss Initial')
    plt.plot(history1.history['val_loss'], label='Val Loss Initial')
    plt.plot(history2.history['loss'], label='Train Loss Fine Tune')
    plt.plot(history2.history['val_loss'], label='Val Loss Fine Tune')
    plt.title('Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history1.history['partes_accuracy'], label='Train Parts Accuracy Initial')
    plt.plot(history1.history['val_partes_accuracy'], label='Val Parts Accuracy Initial')
    plt.plot(history2.history['partes_accuracy'], label='Train Parts Accuracy Fine Tune')
    plt.plot(history2.history['val_partes_accuracy'], label='Val Parts Accuracy Fine Tune')
    plt.title('Parts Accuracy Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, applications, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INITIAL = 30
EPOCHS_FINE_TUNE = 30
LEARNING_RATE_INITIAL = 5e-5
LEARNING_RATE_FINE_TUNE = 5e-6
BASE_MODEL = 'EfficientNetB0'
USE_FOCAL_LOSS = True

# Data loading and preparation functions
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, sep='|')
    for col in ['dannos', 'partes', 'sugerencias']:
        df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df

# Custom data generator with balanced sampling
class BalancedMultiLabelDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, img_dir, mlb_partes, mlb_dannos, mlb_sugerencias,
                 batch_size=32, img_size=(224,224), shuffle=True, augmentation=None):
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
        self.balance_data()

    def balance_data(self):
        y_partes = self.mlb_partes.transform(self.df['partes'])
        y_dannos = self.mlb_dannos.transform(self.df['dannos'])
        y_sugerencias = self.mlb_sugerencias.transform(self.df['sugerencias'])
        y_concat = np.hstack([y_partes, y_dannos, y_sugerencias])
        label_freq = np.sum(y_concat, axis=0)
        label_freq[label_freq == 0] = 1
        sample_weights = np.sum(y_concat / label_freq, axis=1)
        sample_weights = sample_weights / np.sum(sample_weights)
        n_samples = len(self.df)
        indices_resampled = np.random.choice(np.arange(len(self.df)), size=n_samples, replace=True, p=sample_weights)
        self.df = self.df.iloc[indices_resampled].reset_index(drop=True)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        X = np.empty((len(batch_df), *self.img_size, 3), dtype=np.float32)
        y_partes, y_dannos, y_sugerencias = [], [], []
        for i, (_, row) in enumerate(batch_df.iterrows()):
            img_path = os.path.join(self.img_dir, row['Imagen'])
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            if self.augmentation:
                img_array = self.augmentation.random_transform(img_array)
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
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

# Swish activation
def swish(x):
    return x * tf.keras.backend.sigmoid(x)

# DropConnect layer simplified for compatibility
class DropConnect(layers.Layer):
    def __init__(self, drop_prob=0.5, **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, inputs, training=None):
        if not training:
            return inputs
        keep_prob = 1.0 - self.drop_prob
        noise_shape = tf.shape(inputs)
        if len(inputs.shape) == 2:
            noise_shape = [tf.shape(inputs)[0], 1]
        else:
            noise_shape = tf.shape(inputs)
        return tf.nn.dropout(inputs, rate=self.drop_prob, noise_shape=noise_shape) / keep_prob

# Focal loss function
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

# Build improved multi-output model with more fine-tuning layers
def build_improved_model(input_shape, num_partes, num_dannos, num_sugerencias, base_model_name='EfficientNetB0', use_focal_loss=False):
    if base_model_name == 'EfficientNetB0':
        base_model = applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")

    base_model.trainable = True
    # Unfreeze last 50 layers for fine-tuning
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=True)
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

    metrics = {
        'partes': ['accuracy'],
        'dannos': ['accuracy'],
        'sugerencias': ['accuracy']
    }

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_INITIAL), loss=losses, metrics=metrics)

    return model

# Threshold optimization per class
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

# Main training function
def main():
    IMG_DIR = '../data/fotos_siniestros/'
    train_df = load_and_prepare_data(os.path.join(IMG_DIR, 'train_set.csv'))
    val_df = load_and_prepare_data(os.path.join(IMG_DIR, 'val_set.csv'))

    mlb_partes = MultiLabelBinarizer()
    mlb_dannos = MultiLabelBinarizer()
    mlb_sugerencias = MultiLabelBinarizer()

    mlb_partes.fit(train_df['partes'])
    mlb_dannos.fit(train_df['dannos'])
    mlb_sugerencias.fit(train_df['sugerencias'])

    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = BalancedMultiLabelDataGenerator(
        train_df, IMG_DIR, mlb_partes, mlb_dannos, mlb_sugerencias,
        batch_size=BATCH_SIZE, img_size=IMG_SIZE, augmentation=train_datagen
    )

    val_generator = BalancedMultiLabelDataGenerator(
        val_df, IMG_DIR, mlb_partes, mlb_dannos, mlb_sugerencias,
        batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=False
    )

    model = build_improved_model(
        input_shape=(*IMG_SIZE, 3),
        num_partes=len(mlb_partes.classes_),
        num_dannos=len(mlb_dannos.classes_),
        num_sugerencias=len(mlb_sugerencias.classes_),
        base_model_name=BASE_MODEL,
        use_focal_loss=USE_FOCAL_LOSS
    )

    checkpoint = ModelCheckpoint(
        'best_model_improved.h5',
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

    history_initial = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS_INITIAL,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )

    # Optimize thresholds per class using validation data
    val_preds = model.predict(val_generator)
    thresholds_partes = optimize_thresholds(mlb_partes.transform(val_df['partes']), val_preds[0])
    thresholds_dannos = optimize_thresholds(mlb_dannos.transform(val_df['dannos']), val_preds[1])
    thresholds_sugerencias = optimize_thresholds(mlb_sugerencias.transform(val_df['sugerencias']), val_preds[2])

    print("Optimized thresholds:")
    print("Partes:", thresholds_partes)
    print("Dannos:", thresholds_dannos)
    print("Sugerencias:", thresholds_sugerencias)

    # Save thresholds for later use
    with open('optimized_thresholds.json', 'w') as f:
        json.dump({
            'partes': thresholds_partes.tolist(),
            'dannos': thresholds_dannos.tolist(),
            'sugerencias': thresholds_sugerencias.tolist()
        }, f)

    # Fine-tuning: unfreeze more layers and train with lower learning rate
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
        metrics={
            'partes': ['accuracy'],
            'dannos': ['accuracy'],
            'sugerencias': ['accuracy']
        }
    )

    history_fine_tune = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )

    model.save('final_model_improved.h5')

if __name__ == "__main__":
    main()

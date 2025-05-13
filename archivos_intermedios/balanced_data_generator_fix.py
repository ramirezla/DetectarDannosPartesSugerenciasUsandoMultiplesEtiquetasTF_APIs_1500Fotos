import os
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler

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
        # Custom multilabel oversampling by duplicating minority samples
        y_partes = self.mlb_partes.transform(self.df['partes'])
        y_dannos = self.mlb_dannos.transform(self.df['dannos'])
        y_sugerencias = self.mlb_sugerencias.transform(self.df['sugerencias'])
        
        y_concat = np.hstack([y_partes, y_dannos, y_sugerencias])
        
        # Calculate sample weights inversely proportional to label frequency
        label_freq = np.sum(y_concat, axis=0)
        label_freq[label_freq == 0] = 1  # avoid division by zero
        sample_weights = np.sum(y_concat / label_freq, axis=1)
        
        # Normalize weights
        sample_weights = sample_weights / np.sum(sample_weights)
        
        # Number of samples to generate to balance dataset approximately
        n_samples = len(self.df)
        
        # Sample indices with replacement according to weights
        indices_resampled = np.random.choice(np.arange(len(self.df)), size=n_samples, replace=True, p=sample_weights)
        
        # Update dataframe balanced
        self.df = self.df.iloc[indices_resampled]
        self.df.reset_index(drop=True, inplace=True)
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.df.iloc[batch_indices]
        
        X = np.empty((len(batch_df), *self.img_size, 3), dtype=np.float32)
        y_partes = []
        y_dannos = []
        y_sugerencias = []
        
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

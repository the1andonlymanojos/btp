import os
# This line MUST be before the tensorflow import
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# --- Configuration ---
# These should match the settings used during tuning
MERGE_LEVEL = 'simple' 
EPOCHS = 100 # Use a higher epoch count for final training
BATCH_SIZE = 32

# --- Dataset Folders ---
TRAIN_FOLDERS = [
    "Emma Raducanu Court Level Practice 2024 (4K 60FPS)",
    "Emma Raducanu ｜ Court Level Practice [4k 60fps]",
    "Novak Djokovic Volley & Smash Training Court Level View - ATP Tennis Practice",
    "World No.1 Iga Swiatek 2024 Court Level Practice with Caroline Garcia (4K 60FPS)",
    "jsinner_h264",
    "jsinner2_h264"
]

# --- Data Loading ---
def merge_shot_classes(shot_type, level='simple'):
    if level == 'none': return shot_type
    if level == 'simple':
        mappings = {'overhead': 'serve', 'smash': 'serve'}
        return mappings.get(shot_type, shot_type)
    if level == 'coarse':
        mappings = {
            'forehand': 'forehand_group', 'forehand_volley': 'forehand_group',
            'backhand': 'backhand_group', 'backhand_slice': 'backhand_group', 'backhand_volley': 'backhand_group',
            'serve': 'serve_group', 'overhead': 'serve_group', 'smash': 'serve_group'
        }
        return mappings.get(shot_type, shot_type)
    return shot_type

def load_data_from_folders(folders, base_path="dataset/", merge_level='simple'):
    X, y = [], []
    print(f"Loading data with merge level: '{merge_level}'")
    for folder in folders:
        dataset_path = os.path.join(base_path, folder)
        if not os.path.exists(dataset_path):
            print(f"Warning: {dataset_path} doesn't exist, skipping...")
            continue
        
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        for shot_csv in tqdm(sorted(csv_files), desc=f"Processing {folder}", unit="file"):
            try:
                data = pd.read_csv(os.path.join(dataset_path, shot_csv))
                name_parts = shot_csv.replace('.csv', '').split('_')
                shot_type = f"{name_parts[0]}_{name_parts[1]}" if len(name_parts) >= 3 and name_parts[1] in ['volley', 'slice'] else name_parts[0]
                shot_type = merge_shot_classes(shot_type, level=merge_level)
                features = data.loc[:, data.columns != 'shot']
                if features.shape[0] == 30:
                    X.append(features.to_numpy())
                    y.append(shot_type)
            except Exception as e:
                print(f"Could not process file {shot_csv}: {e}")
            
    return np.stack(X, axis=0), np.array(y)

# --- Model Definition ---
# This function defines your best model architecture
def create_best_model(input_shape, num_classes, params):
    """
    Creates the Simple_GRU model with the best hyperparameters found during tuning.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(units=params['units'], dropout=params['dropout']),
        layers.Dense(units=params['units'] // 2, activation='relu'),
        layers.Dense(units=num_classes, activation='softmax')
    ])
    optimizer = keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser(description="Train and save the best shot classification model.")
    parser.add_argument('--output_path', type=str, default='final_shot_classifier.keras', help='Path to save the final trained model.')
    args = parser.parse_args()

    # --- Best Hyperparameters (from your tuning results) ---
    best_params = {
        'units': 128, 
        'dropout': 0.2, 
        'learning_rate': 0.001 # Assuming this was the rate, adjust if needed
    }
    print(f"Using best hyperparameters for Simple_GRU: {best_params}")

    # 1. Load the entire dataset
    X_train, y_train_str = load_data_from_folders(TRAIN_FOLDERS, merge_level=MERGE_LEVEL)
    
    # 2. Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_str)
    num_classes = len(le.classes_)
    y_train_cat = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
    
    print("\nClass mapping:")
    for i, class_name in enumerate(le.classes_):
        print(f"  {i}: {class_name}")

    # 3. Compute class weights for the full dataset
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights: {class_weight_dict}")

    # 4. Build the model
    input_shape = X_train.shape[1:]
    model = create_best_model(input_shape, num_classes, best_params)
    model.summary()

    # 5. Train the model on the ENTIRE dataset
    print(f"\nTraining final model for {EPOCHS} epochs...")
    model.fit(X_train, y_train_cat,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              class_weight=class_weight_dict,
              callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)],
              verbose=1)

    # 6. Save the final model
    model.save(args.output_path)
    print(f"\n--- Model training complete. ---")
    print(f"Final model saved successfully to: {args.output_path}")


if __name__ == "__main__":
    main()
f
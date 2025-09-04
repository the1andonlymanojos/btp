import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import random
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import json

# --- Configuration ---
MERGE_SIMILAR_SHOTS = True
N_SPLITS = 3  # Folds for cross-validation
EPOCHS = 50   # Max epochs for training
BATCH_SIZE = 32
level=True
N_ITER_RANDOM_SEARCH = 8 # Number of random combinations to test per model

# --- Dataset Folders ---
TRAIN_FOLDERS = [
     "emma1clean/Emma Raducanu Court Level Practice 2024 (4K 60FPS)",
    "emma2clean/Emma Raducanu ｜ Court Level Practice [4k 60fps]",
    "novaksmashclean/novaksmashclean", 
    "World No.1 Iga Swiatek 2024 Court Level Practice with Caroline Garcia (4K 60FPS)",
    "jsinner_h264",
    "jsinner2_h264",
     "Novak Djokovic Volley & Smash Training Court Level View - ATP Tennis Practice",
    "2",
    "3",
    "4",
    "5",
    "6",
    "8",
    "9"

]

# --- Data Loading ---
def merge_shot_classes(shot_type, merge_similar=True):
    if not merge_similar: return shot_type
    shot_mappings = {'overhead': 'serve', 'smash': 'serve'}
    if level:
        shot_mappings = {
            # Group all forehands
            'forehand': 'forehand_group',
            'forehand_volley': 'forehand_group',
            
            # Group all backhands
            'backhand': 'backhand_group',
            'backhand_slice': 'backhand_group',
            'backhand_volley': 'backhand_group',
            
            # Group serves and overheads
            'serve': 'serve_group',
            'overhead': 'serve_group',
            'smash': 'serve_group'
            
            # 'neutral' remains 'neutral'
        }
        return shot_mappings.get(shot_type, shot_type)
    return shot_mappings.get(shot_type, shot_type)

def load_data_from_folders(folders, base_path="dataset/"):
    X, y = [], []
    print("Loading data...")
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
                shot_type = merge_shot_classes(shot_type, MERGE_SIMILAR_SHOTS)
                features = data.loc[:, data.columns != 'shot']
                if features.shape[0] == 30:
                    X.append(features.to_numpy())
                    y.append(shot_type)
            except Exception as e:
                print(f"Could not process file {shot_csv}: {e}")
            
    return np.stack(X, axis=0), np.array(y)

# --- Model Definitions ---
# Each function now accepts its specific hyperparameters.

def create_simple_gru_model(input_shape, num_classes, units=32, dropout=0.2, learning_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=input_shape), layers.GRU(units=units, dropout=dropout),
        layers.Dense(units=units // 2, activation='relu'), layers.Dense(units=num_classes, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_simple_lstm_model(input_shape, num_classes, units=32, dropout=0.2, learning_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=input_shape), layers.LSTM(units=units, dropout=dropout),
        layers.Dense(units=units // 2, activation='relu'), layers.Dense(units=num_classes, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_bidirectional_lstm_model(input_shape, num_classes, units=32, dropout=0.2, learning_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=input_shape), layers.Bidirectional(layers.LSTM(units=units, dropout=dropout)),
        layers.Dense(units=units // 2, activation='relu'), layers.Dense(units=num_classes, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_model(input_shape, num_classes, filters=64, kernel_size=3, dropout=0.3, learning_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=input_shape), layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        layers.Dropout(dropout), layers.Conv1D(filters=filters//2, kernel_size=kernel_size, activation='relu'),
        layers.GlobalMaxPooling1D(), layers.Dense(units=filters//4, activation='relu'),
        layers.Dense(units=num_classes, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_gru_hybrid_model(input_shape, num_classes, filters=64, kernel_size=3, gru_units=32, dropout=0.3, learning_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=input_shape), layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2), layers.Dropout(dropout),
        layers.GRU(units=gru_units, dropout=dropout), layers.Dense(units=gru_units//2, activation='relu'),
        layers.Dense(units=num_classes, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_bilstm_attention_model(input_shape, num_classes, units=32, dropout=0.2, learning_rate=0.001):
    """A Bidirectional LSTM model with a self-attention mechanism."""
    inputs = layers.Input(shape=input_shape)
    # BiLSTM layer needs to return the full sequence for attention
    lstm_out = layers.Bidirectional(layers.LSTM(units, dropout=dropout, return_sequences=True))(inputs)
    
    # Attention layer
    # For self-attention, the query and value are the same (the BiLSTM output)
    attention_out = layers.Attention()([lstm_out, lstm_out])
    
    # Condense the output of the attention layer before the final classifier
    context_vector = layers.GlobalAveragePooling1D()(attention_out)
    
    # Classifier head
    dense_out = layers.Dense(units // 2, activation='relu')(context_vector)
    outputs = layers.Dense(num_classes, activation='softmax')(dense_out)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Hyperparameter Search Definitions ---

def get_model_definitions():
    """Returns a dictionary mapping model names to their creation functions and parameter distributions."""
    
    # Define parameter search space for each model
    rnn_params = {
        'units': [32, 64, 128],
        'dropout': [0.2, 0.3, 0.4],
        'learning_rate': [1e-3, 5e-4]
    }
    cnn_params = {
        'filters': [32, 64],
        'kernel_size': [3, 5],
        'dropout': [0.3, 0.5],
        'learning_rate': [1e-3, 5e-4]
    }
    hybrid_params = {
        'filters': [32, 64],
        'kernel_size': [3, 5],
        'gru_units': [32, 64],
        'dropout': [0.3, 0.5],
        'learning_rate': [1e-3, 5e-4]
    }

    return {
        'Simple_GRU': (create_simple_gru_model, rnn_params),
        'Simple_LSTM': (create_simple_lstm_model, rnn_params),
        'BiLSTM': (create_bidirectional_lstm_model, rnn_params),
        'BiLSTM_Attention': (create_bilstm_attention_model, rnn_params), # New model
        '1D_CNN': (create_cnn_model, cnn_params),
        'CNN_GRU_Hybrid': (create_cnn_gru_hybrid_model, hybrid_params)
    }

# --- Evaluation Core ---
def run_single_evaluation(X, y, model_builder, params):
    """Runs a single k-fold evaluation for a given model and its hyperparameters."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    input_shape = X.shape[1:]
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_accuracies = []

    for _ in skf.split(X, y_encoded):
        train_idx, val_idx = _
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))
        
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
        
        model = model_builder(input_shape, num_classes, **params)
        
        model.fit(X_train, y_train_cat, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  class_weight=class_weight_dict, validation_data=(X_val, y_val_cat),
                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)],
                  verbose=0)
        
        _, accuracy = model.evaluate(X_val, y_val_cat, verbose=0)
        fold_accuracies.append(accuracy)
        
    return np.mean(fold_accuracies)

def random_search(X, y, model_builder, param_dist, n_iter):
    """Performs a random search for a given model."""
    best_score = -1
    best_params = None
    
    for i in range(n_iter):
        params = {key: random.choice(val) for key, val in param_dist.items()}
        
        print(f"  Iter {i+1}/{n_iter} | Testing params: {params}")
        score = run_single_evaluation(X, y, model_builder, params)
        print(f"  -> Mean Accuracy: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_params = params
            
    return best_params, best_score

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Automated multi-model hyperparameter tuning.")
    parser.add_argument('--data_path', type=str, default='dataset/', help='Path to the root dataset directory.')
    parser.add_argument('--model', type=str, default='all', help='Specify a single model to tune, or "all".')
    
    args = parser.parse_args()
    
    X, y = load_data_from_folders(TRAIN_FOLDERS, base_path=args.data_path)
    print(f"\nData loaded. Shapes: X={X.shape}, y={y.shape}")
    
    all_model_definitions = get_model_definitions()
    models_to_tune = list(all_model_definitions.keys())

    if args.model != 'all' and args.model in all_model_definitions:
        models_to_tune = [args.model]
    elif args.model != 'all':
        print(f"Error: Model '{args.model}' not found. Tuning all models.")

    results = []
    os.makedirs('optimised', exist_ok=True)
    for model_name in models_to_tune:
        print(f"\n--- Tuning Model: {model_name} ---")
        model_builder, param_dist = all_model_definitions[model_name]
        
        best_params, best_score = random_search(X, y, model_builder, param_dist, N_ITER_RANDOM_SEARCH)
        
        # Retrain best model on full data and save
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        num_classes = len(le.classes_)
        input_shape = X.shape[1:]
        y_cat = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
        best_model = model_builder(input_shape, num_classes, **best_params)
        best_model.fit(X, y_cat, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        model_path = os.path.join('optimised', f'{model_name}_best_model.h5')
        best_model.save(model_path)
        print(f"Saved best {model_name} model to {model_path}")
        
        results.append({
            'Model': model_name,
            'Best Score': best_score,
            'Best Params': best_params
        })
        print(f"--- Best result for {model_name}: Score={best_score:.4f}, Params={best_params} ---")

    print("\n\n--- Automated Hyperparameter Tuning Complete ---")
    summary_df = pd.DataFrame(results).sort_values(by='Best Score', ascending=False).set_index('Model')
    print(summary_df[['Best Score']])
    print("\nFull results with best parameters:")
    print(summary_df)

    # Save results to a text file and JSON for easy parsing
    results_txt_path = os.path.join('optimised', 'results.txt')
    results_json_path = os.path.join('optimised', 'results.json')
    with open(results_txt_path, 'w') as f:
        f.write(summary_df.to_string())
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_txt_path} and {results_json_path}")


if __name__ == "__main__":
    main()
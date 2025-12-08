import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# --- IMPORTS BOOSTING LIBRARIES ---
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Metric
from sklearn.metrics import mean_absolute_percentage_error

# ==========================================
# KONFIGURASI
# ==========================================
DATA_FILE = 'dataset.csv'
TARGET_GAME = "Counter-Strike 2"
LOOK_BACK = 1       # Lag 1 hari
TEST_SIZE = 30      # 30 Hari terakhir

# ==========================================
# 1. PERSIAPAN DATA
# ==========================================
def get_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} tidak ditemukan.")
        return None, None, None, None, None

    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Ambil data Close
    series = df['Close'].values.astype('float32')
    
    # Time Series Transformation
    X, y = [], []
    for i in range(len(series) - LOOK_BACK):
        X.append(series[i : i + LOOK_BACK])
        y.append(series[i + LOOK_BACK])
        
    X = np.array(X)
    y = np.array(y)
    
    train_size = len(X) - TEST_SIZE
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return df, X_train, y_train, X_test, y_test

# ==========================================
# 2. DEFINISI MODEL (DITAMBAH DCLGM)
# ==========================================
def get_boosting_models():
    models = {}
    
    # --- A. STANDARD BOOSTING ---
    models['AdaBoost'] = AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    models['Gradient Boosting'] = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    
    # --- B. THE BIG THREE ---
    # XGBoost Standard (Gbtree)
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, n_jobs=-1, objective='reg:squarederror')
    models['XGBoost'] = xgb_model
    
    # LightGBM Standard
    lgbm_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, verbose=-1)
    models['LightGBM'] = lgbm_model
    
    # CatBoost Standard
    cat_model = cb.CatBoostRegressor(iterations=200, learning_rate=0.05, depth=4, verbose=0, loss_function='RMSE')
    models['CatBoost'] = cat_model

    # --- C. SPECIAL VARIANTS ---
    # XGBoost DART (Dropout - bagus untuk mencegah overfitting)
    xgb_dart = xgb.XGBRegressor(booster='dart', rate_drop=0.1, skip_drop=0.5, 
                                n_estimators=200, learning_rate=0.05, max_depth=4, n_jobs=-1)
    models['XGBoost (DART)'] = xgb_dart

    # --- D. DCLGM (Dart-Cat-LGBM Ensemble) ---
    # Gabungan model-model terbaik
    estimators = [
        ('dart', xgb_dart),
        ('cat', cat_model),
        ('lgbm', lgbm_model)
    ]
    # Voting Regressor merata-rata prediksi dari ketiganya
    models['DCLGM (Ensemble)'] = VotingRegressor(estimators=estimators)

    return models

# ==========================================
# 3. EKSEKUSI
# ==========================================
def main():
    print("--- MEMUAT DATA ---")
    df, X_train, y_train, X_test, y_test = get_data()
    if df is None: return
    
    models = get_boosting_models()
    results = []
    predictions = {}
    
    print(f"\n--- TRAINING {len(models)} BOOSTING MODELS ---")
    
    for name, model in tqdm(models.items()):
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions[name] = pred
            
            mape = mean_absolute_percentage_error(y_test, pred) * 100
            results.append({'Model': name, 'MAPE (%)': mape})
            
        except Exception as e:
            print(f"Gagal pada {name}: {e}")

    # Urutkan Hasil
    df_res = pd.DataFrame(results).sort_values('MAPE (%)')
    
    print("\n=== PERINGKAT AKURASI (MAPE Terkecil) ===")
    print(df_res)

    # Simpan Hasil CSV
    df_res.to_csv('hasil_benchmark_boosting.csv', index=False)
    
    # ==========================================
    # 4. VISUALISASI
    # ==========================================
    test_dates = df['Date'].iloc[-len(y_test):]
    
    # Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x='MAPE (%)', y='Model', data=df_res, palette='magma')
    plt.title('Boosting Benchmark: Siapa Paling Akurat?')
    plt.tight_layout()
    plt.savefig('boosting_dclgm_bar.png')
    
    # Line Chart
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test, label='ACTUAL DATA', color='black', linewidth=3, linestyle='--')
    
    # Plot Top 3 Models + DCLGM
    top_models = df_res.head(3)['Model'].tolist()
    if 'DCLGM (Ensemble)' not in top_models:
        top_models.append('DCLGM (Ensemble)')
        
    for name in top_models:
        plt.plot(test_dates, predictions[name], label=name)

    plt.title(f'Prediction: {TARGET_GAME}')
    plt.xlabel('Tanggal')
    plt.ylabel('Jumlah Pemain')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('boosting_dclgm_line.png')

    print("\nSelesai! Cek 'boosting_dclgm_bar.png' dan 'boosting_dclgm_line.png'.")

if __name__ == "__main__":
    main()
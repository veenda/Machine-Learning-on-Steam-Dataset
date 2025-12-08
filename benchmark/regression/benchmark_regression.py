import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# --- IMPORTS SKLEARN ---
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.compose import TransformedTargetRegressor

# Linear & Regularized Models
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, 
                                  HuberRegressor, QuantileRegressor, SGDRegressor,
                                  PoissonRegressor, GammaRegressor, TweedieRegressor)

# Support Vector & Tree
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

# --- KONFIGURASI ---
DATA_FILE = 'dataset.csv'
TARGET_GAME = "Counter-Strike 2"
LOOK_BACK = 1        # Time Series Regression biasanya menggunakan lag 1 (hari sebelumnya)
TEST_SIZE = 30       # 30 Data terakhir untuk testing

# ==========================================
# 1. PERSIAPAN DATA
# ==========================================
def get_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: File {DATA_FILE} tidak ditemukan. Jalankan convert_data.py dulu!")
        return None, None, None, None

    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Ambil nilai Close (Players Now)
    # Kita pastikan nilainya float dan > 0 untuk model Log/Poisson
    dataset = df['Close'].values.reshape(-1, 1).astype('float32')
    # Tambahkan epsilon kecil agar tidak ada log(0) error
    dataset = np.maximum(dataset, 1.0) 
    
    # Split Train/Test (Tanpa Shuffle karena Time Series)
    train_size = len(dataset) - TEST_SIZE
    train, test = dataset[0:train_size], dataset[train_size:]
    
    # --- FORMING LAG FEATURES (X=Hari Kemarin, Y=Hari Ini) ---
    def create_lags(data, look_back=1):
        X, Y = [], []
        for i in range(len(data) - look_back):
            a = data[i:(i + look_back), 0]
            X.append(a)
            Y.append(data[i + look_back, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_lags(train, LOOK_BACK)
    X_test, y_test = create_lags(test, LOOK_BACK)

    return df, X_train, y_train, X_test, y_test

# ==========================================
# 2. DEFINISI SEMUA MODEL
# ==========================================
def get_models():
    models = {}

    # 1. Linear Regression (Standard)
    models['Linear Regression'] = LinearRegression()

    # 2. Polynomial Regression (Degree 3)
    models['Polynomial Regression'] = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

    # 3. Quadratic Regression (Degree 2)
    models['Quadratic Regression'] = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

    # 4. Quantile Regression
    # Sklearn QuantileRegressor cukup lambat untuk data besar, alternatifnya GradientBoosting dengan loss quantile
    models['Quantile Regression'] = GradientBoostingRegressor(loss='quantile', alpha=0.5)

    # 5. Elastic Net Regression
    models['Elastic Net'] = ElasticNet(alpha=0.1, l1_ratio=0.5)

    # 6. Principal Components Regression (PCR)
    models['PCR'] = make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())

    # 7. Partial Least Squares Regression (PLSR)
    models['PLS Regression'] = PLSRegression(n_components=1)

    # 8. Support Vector Regression (SVR)
    models['SVR'] = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))

    # 9. Poisson Regression (Count Data)
    models['Poisson Regression'] = PoissonRegressor()

    # 10. Negative Binomial Regression (Proxy: Gamma/Tweedie p=2)
    # Negative Binomial tidak ada native di sklearn, Gamma sering dipakai sebagai proxy overdispersion
    models['Negative Binomial (Proxy)'] = GammaRegressor()

    # 11. Quasi Poisson Regression (Proxy: Tweedie p=1.5)
    # Tweedie dengan power 1-2 bertindak seperti Compound Poisson-Gamma (Quasi Poisson)
    models['Quasi Poisson (Proxy)'] = TweedieRegressor(power=1.5)

    # 12. Ridge Regression
    models['Ridge Regression'] = Ridge(alpha=1.0)

    # 13. Lasso Regression
    models['Lasso Regression'] = Lasso(alpha=0.1)

    # 14. Time Series Regression
    # Ini sebenarnya konsep, bukan satu model spesifik.
    # Kita gunakan Linear Regression dengan Lag Features yang sudah kita buat (X_train)
    models['Time Series Regression (AR)'] = LinearRegression()

    # 15. Decision Tree Regression
    models['Decision Tree'] = DecisionTreeRegressor(max_depth=5)

    # 16. Huber Regression (Robust to Outliers)
    models['Huber Regression'] = make_pipeline(StandardScaler(), HuberRegressor())

    # --- CUSTOM TRANSFORMATIONS (Power, Exp, Log) ---
    
    # 17. Power Regression: Y = a * X^b  => log(Y) = log(a) + b*log(X)
    # Kita transformasi Input X jadi Log(X) dan Target Y jadi Log(Y)
    models['Power Regression'] = make_pipeline(
        FunctionTransformer(np.log1p, validate=True), # Log X
        TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1) # Log Y
    )

    # 18. Exponential Regression: Y = a * e^(bX) => log(Y) = log(a) + bX
    # Kita hanya transformasi Target Y jadi Log(Y)
    models['Exponential Regression'] = TransformedTargetRegressor(
        regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1
    )

    # 19. Logarithmic Regression: Y = a + b*ln(X)
    # Kita hanya transformasi Input X jadi Log(X)
    models['Logarithmic Regression'] = make_pipeline(
        FunctionTransformer(np.log1p, validate=True),
        LinearRegression()
    )

    return models

# ==========================================
# 3. EKSEKUSI UTAMA
# ==========================================
def main():
    print("--- MEMUAT DATA ---")
    df, X_train, y_train, X_test, y_test = get_data()
    if df is None: return

    # Persiapan Visualisasi
    test_dates = df['Date'].iloc[-len(y_test):]
    
    print(f"\n--- MELATIH {len(get_models())} MODEL REGRESI ---")
    
    results = []
    predictions = {}
    
    models = get_models()
    
    # Loop Training
    for name, model in tqdm(models.items()):
        try:
            # Fit Model
            model.fit(X_train, y_train)
            
            # Predict
            pred = model.predict(X_test)
            pred = pred.flatten() # Pastikan 1D array
            
            predictions[name] = pred
            
            # Hitung Error (MAPE)
            # MAPE = Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
            results.append({'Model': name, 'MAPE (%)': mape})
            
        except Exception as e:
            print(f"Gagal pada model {name}: {e}")

    # Urutkan Hasil
    df_results = pd.DataFrame(results).sort_values('MAPE (%)')
    
    print("\n--- PERINGKAT AKURASI (Error Terkecil Menang) ---")
    print(df_results)
    
    # Simpan Hasil CSV
    df_results.to_csv('hasil_benchmark_regresi.csv', index=False)

    # ==========================================
    # 4. VISUALISASI
    # ==========================================
    
    # Grafik 1: Bar Chart Perbandingan Error
    plt.figure(figsize=(12, 8))
    sns.barplot(x='MAPE (%)', y='Model', data=df_results, palette='coolwarm')
    plt.title('Perbandingan Error Model (Semakin Kecil Semakin Bagus)')
    plt.xlabel('Rata-rata Error (%)')
    plt.tight_layout()
    plt.savefig('grafik_bar_regresi.png')
    
    # Grafik 2: Line Chart (Top 5 vs Data Asli)
    plt.figure(figsize=(14, 7))
    
    # Plot Data Asli
    plt.plot(test_dates, y_test, label='Data Asli (Actual)', color='black', linewidth=2.5, linestyle='-')
    
    # Plot Top 5 Model
    top_5 = df_results.head(5)['Model'].tolist()
    for name in top_5:
        plt.plot(test_dates, predictions[name], label=name, linestyle='--')
        
    plt.title(f'Prediksi vs Realita (Top 5 Model Terbaik) - {TARGET_GAME}')
    plt.xlabel('Tanggal')
    plt.ylabel('Jumlah Pemain')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('grafik_line_regresi.png')

    print("\nSelesai! Cek file 'grafik_bar_regresi.png' dan 'grafik_line_regresi.png'.")
    print("Catatan: Logistic, Ordinal, dan Cox Regression dilewati karena tidak kompatibel dengan prediksi angka kontinu.")

if __name__ == "__main__":
    main()
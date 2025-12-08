import json
import pandas as pd
import os

# --- KONFIGURASI ---
INPUT_FILE = 'merged_lines.json'  # Nama file JSON Anda
OUTPUT_FILE = 'dataset.csv'       # Nama file CSV output
TARGET_GAME = "Counter-Strike 2"  # Game yang ingin diprediksi

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File {INPUT_FILE} tidak ditemukan!")
        return

    print(f"Membaca file {INPUT_FILE}...")
    data = []
    
    # Membaca file line by line karena format JSON Lines
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                # Filter hanya game yang kita inginkan
                if entry.get('name') == TARGET_GAME:
                    data.append(entry)
            except json.JSONDecodeError:
                continue

    if not data:
        print(f"Tidak ada data ditemukan untuk game: {TARGET_GAME}")
        return

    # Membuat DataFrame
    df = pd.DataFrame(data)
    
    # Mengubah timestamp menjadi tanggal yang bisa dibaca
    df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Mengurutkan berdasarkan waktu (SANGAT PENTING untuk Time Series)
    df = df.sort_values('Date')
    
    # Memilih kolom yang relevan dan mengganti nama agar mirip data saham
    # players_now -> Close (Harga penutupan)
    df_export = df[['Date', 'players_now']].copy()
    df_export.columns = ['Date', 'Close']
    
    # Menghapus duplikat waktu jika ada
    df_export = df_export.drop_duplicates(subset=['Date'])
    
    # Menyimpan ke CSV
    df_export.to_csv(OUTPUT_FILE, index=False)
    print(f"Sukses! Data {len(df_export)} baris telah disimpan ke '{OUTPUT_FILE}'.")
    print(f"Rentang waktu: {df_export['Date'].min()} sampai {df_export['Date'].max()}")

if __name__ == "__main__":
    main()
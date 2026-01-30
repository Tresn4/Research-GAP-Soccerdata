import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

print("Data Load")

def load_fbref_safe(filename):
    try:
        df = pd.read_csv(filename, header=1)
    except: return pd.DataFrame()
    
    cols = df.columns.tolist()
    rename_map = {}
    if 'Unnamed: 0' in cols: rename_map['Unnamed: 0'] = 'League'
    if 'Unnamed: 1' in cols: rename_map['Unnamed: 1'] = 'Season'
    if 'Unnamed: 2' in cols: rename_map['Unnamed: 2'] = 'Team'
    df.rename(columns=rename_map, inplace=True)
    
    # Hapus baris header berulang
    if 'Season' in df.columns:
        df = df[df['Season'] != 'season']
        df['Season'] = pd.to_numeric(df['Season'], errors='coerce')
        
    return df

# Load File
df_std = load_fbref_safe('epl_team_standard.csv')
df_time = load_fbref_safe('epl_team_playing_time.csv')
df_misc = load_fbref_safe('epl_team_misc.csv')

# Gabungkan Data
def clean_numeric(df, cols):
    if df.empty: return df
    available = [c for c in cols if c in df.columns]
    df_sub = df[['Team', 'Season'] + available].copy()
    for c in available: df_sub[c] = pd.to_numeric(df_sub[c], errors='coerce')
    return df_sub

df_std_clean = clean_numeric(df_std, ['xG', 'Gls', 'Ast'])
df_time_clean = clean_numeric(df_time, ['PPM', 'MP'])
df_misc_clean = clean_numeric(df_misc, ['Int', 'TklW'])

df_final = df_std_clean
if not df_time_clean.empty: df_final = pd.merge(df_final, df_time_clean, on=['Team', 'Season'], how='inner')
if not df_misc_clean.empty: df_final = pd.merge(df_final, df_misc_clean, on=['Team', 'Season'], how='inner')

# Target Poin
df_final['Total_Points'] = df_final['PPM'] * df_final['MP']
df_final['Total_Points'] = df_final['Total_Points'].round().fillna(0).astype(int)

train_seasons = [2021, 2122, 2223]
test_season = 2324

train_data = df_final[df_final['Season'].isin(train_seasons)]
test_data = df_final[df_final['Season'] == test_season]

features = ['xG', 'Gls', 'Ast', 'Int'] 
target = 'Total_Points'

if not train_data.empty and not test_data.empty:
    # Training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_data[features], train_data[target])
    
    # Testing
    y_test = test_data[target]
    y_pred = model.predict(test_data[features])
    
    # Hitung MAE
    mae = mean_absolute_error(y_test, y_pred)
    
    # Buat DataFrame Hasil
    res = pd.DataFrame({
        'Team': test_data['Team'],
        'Actual': y_test,
        'Predicted': y_pred
    })
    res['Error'] = abs(res['Actual'] - res['Predicted'])
    res_sorted = res.sort_values('Error', ascending=False)

    # OUTPUT 1: TABEL 
    print(f"ANALISIS PREDIKSI: PREMIER LEAGUE MUSIM 2023/2024")
    print("-" * 60)
    print(f"MAE (Rata-rata Error): {mae:.2f} Poin")
    print("-" * 60)
    print(res_sorted.to_string(index=False))
    print("-" * 60)

    # OUTPUT 2: SCATTER PLOT
    plt.figure(figsize=(10, 7))
    
    # Plot Titik-Titik Semakin Merah = Semakin Salah prediksinya
    scatter = sns.scatterplot(
        data=res, 
        x='Actual', 
        y='Predicted', 
        hue='Error', 
        palette='coolwarm', 
        s=120,
        edgecolor='black' 
    )
    
    # Garis Ideal/Prediksi Sempurna
    min_val = min(res['Actual'].min(), res['Predicted'].min()) - 5
    max_val = max(res['Actual'].max(), res['Predicted'].max()) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Prediksi Sempurna (Target)')
    
    texts = []
    for i in range(5): # Ambil top 5 error terbesar
        row = res_sorted.iloc[i]
        plt.text(
            row['Actual'] + 1, 
            row['Predicted'] - 2, 
            f"{row['Team']}\n(Err: {row['Error']:.1f})", 
            fontsize=9, 
            fontweight='bold', 
            color='darkred'
        )

    plt.title(f'Visualisasi Kegagalan Prediksi Klasemen di Musim 2024\n(MAE: {mae:.2f} Poin)', fontsize=14)
    plt.xlabel('Poin Asli', fontsize=12)
    plt.ylabel('Prediksi Model (Random Forest)', fontsize=12)
    plt.legend(title='Besar Error')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Simpan Gambar
    filename_img = 'scatter_plot_pl.png'
    plt.savefig(filename_img, dpi=300)
    print(f"\n Visualisasi Scatter Plot disimpan sebagai: '{filename_img}'")
    plt.show()

else:
    print("ERROR! Data tidak ditemukan. Cek nama file CSV.")
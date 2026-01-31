import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os

FILENAME = 'epl_final.csv'

# 1. LOAD DATASET
if not os.path.exists(FILENAME):
    print(f"[ERROR] File '{FILENAME}' tidak ditemukan.")
    exit()

try:
    df = pd.read_csv(FILENAME, encoding='latin1')
    
    # Standarisasi Nama Kolom
    rename_map = {
        'MatchDate': 'Date',
        'FullTimeHomeGoals': 'FTHG',
        'FullTimeAwayGoals': 'FTAG',
        'FullTimeResult': 'FTR'
    }
    df.rename(columns=rename_map, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
except Exception as e:
    print(f"Error membaca file: {e}")
    exit()

# 2. PERSIAPAN DATA (FEATURE ENGINEERING)
# Target: Musim 2023/24
# Training: 4 Musim sebelumnya
target_season = '2023/24'
train_seasons = ['2019/20', '2020/21', '2021/22', '2022/23']
all_seasons = train_seasons + [target_season]

df_filtered = df[df['Season'].isin(all_seasons)].copy()

def calculate_standings(match_data):
    teams = pd.unique(match_data[['HomeTeam', 'AwayTeam']].values.ravel('K'))
    teams = [t for t in teams if str(t) != 'nan']
    
    standings = []
    for team in teams:
        home = match_data[match_data['HomeTeam'] == team]
        away = match_data[match_data['AwayTeam'] == team]
        
        wins = len(home[home['FTR'] == 'H']) + len(away[away['FTR'] == 'A'])
        draws = len(home[home['FTR'] == 'D']) + len(away[away['FTR'] == 'D'])
        losses = len(home[home['FTR'] == 'A']) + len(away[away['FTR'] == 'H'])
        gf = home['FTHG'].sum() + away['FTAG'].sum()
        ga = home['FTAG'].sum() + away['FTHG'].sum()
        points = (wins * 3) + (draws * 1)
        
        standings.append({
            'Team': team, 
            'Wins': wins, 'Draws': draws, 'Losses': losses,
            'GF': gf, 'GA': ga, 
            'Points': points
        })
    return pd.DataFrame(standings)

# Generate Tabel Klasemen
final_data = []
for s in all_seasons:
    matches = df_filtered[df_filtered['Season'] == s]
    if not matches.empty:
        tbl = calculate_standings(matches)
        tbl['Season'] = s
        final_data.append(tbl)

df_final = pd.concat(final_data, ignore_index=True)

# 3. TRAINING MODEL (RANDOM FOREST)
train_data = df_final[df_final['Season'].isin(train_seasons)]
test_data = df_final[df_final['Season'] == target_season]

# Fitur Prediksi: Fokus pada Produktivitas Gol (Simulasi Model Konvensional)
features = ['GF', 'GA'] 
target = 'Points'

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(train_data[features], train_data[target])
preds = rf.predict(test_data[features])

# 4. OUTPUT HASIL (TABEL PERBANDINGAN)
results = test_data[['Team', 'Points']].copy()
results['Predicted'] = preds
results['Error'] = abs(results['Points'] - results['Predicted'])
mae = mean_absolute_error(results['Points'], results['Predicted'])



print(f"TABEL HASIL PREDIKSI FINAL STANDINGS - Musim {target_season})")
print("-" * 60)
print(f"{'Team':<20} | {'Actual':<8} | {'Predicted':<10} | {'Error':<8}")
print("-" * 60)

# Sortir berdasarkan Error terbesar
results_sorted = results.sort_values('Error', ascending=False)

for index, row in results_sorted.iterrows():
    print(f"{row['Team']:<20} | {row['Points']:<8} | {row['Predicted']:<10.2f} | {row['Error']:<8.2f}")

print("-" * 60)
print(f"Rata-rata Error (MAE): {mae:.2f} Poin")
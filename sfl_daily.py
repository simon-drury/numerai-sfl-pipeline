import numpy as np
import pandas as pd
import json
from datetime import datetime
from numerapi import NumerAPI
import os
from collections import defaultdict
import sys
import pyarrow.parquet as pq

# --- CONFIGURATION ---
DATE_STR = datetime.now().strftime('%Y%m%d')
DIR_V5 = "v5.1"
FILE_TRAIN = os.path.join(DIR_V5, "train.parquet")
FILE_LIVE = os.path.join(DIR_V5, "live.parquet")
FILE_FEATURES = os.path.join(DIR_V5, "features.json")
ROUND_TRACKER = os.path.join(DIR_V5, "round_version.txt")

print(f"### SFL STRICT SYNC MASTER [{DATE_STR}] ###")
print("=" * 60)

# ---------------------------------------------------------
# 1. STRICT DATA SYNC (ROUND NUMBER CHECK)
# ---------------------------------------------------------
napi = NumerAPI()
os.makedirs(DIR_V5, exist_ok=True)

print(">> 1. CHECKING LIVE ROUND STATUS...")
try:
    current_round = napi.get_current_round()
    print(f"   API reports Current Round: {current_round}")
    
    local_round = 0
    if os.path.exists(ROUND_TRACKER):
        with open(ROUND_TRACKER, 'r') as f:
            try:
                local_round = int(f.read().strip())
            except:
                local_round = 0
    
    if current_round > local_round or not os.path.exists(FILE_LIVE):
        print(f"   [UPDATE REQUIRED] Local: {local_round} vs Live: {current_round}")
        print("   Downloading fresh V5.1 datasets...")
        
        napi.download_dataset("v5.1/train.parquet", FILE_TRAIN)
        napi.download_dataset("v5.1/live.parquet", FILE_LIVE)
        napi.download_dataset("v5.1/features.json", FILE_FEATURES)
        
        with open(ROUND_TRACKER, 'w') as f:
            f.write(str(current_round))
        print("   Download Complete. Round synced.")
    else:
        print(f"   [READY] Local data matches Live Round {current_round}.")
        
except Exception as e:
    print(f"   ! CRITICAL API FAILURE: {e}")
    if not os.path.exists(FILE_LIVE):
        print("   ! NO LOCAL DATA. EXITING.")
        sys.exit(1)
    print("   ! WARNING: Proceeding with existing local files (Risk of stale data).")

# ---------------------------------------------------------
# 2. LOAD DATA (LOW MEMORY STREAMING)
# ---------------------------------------------------------
print(">> 2. LOADING SUBSTRATE...")

with open(FILE_FEATURES, 'r') as f:
    feature_names = json.load(f)['feature_sets']['medium'][:50]

try:
    pf = pq.ParquetFile(FILE_TRAIN)
    train = next(pf.iter_batches(batch_size=25000, columns=feature_names + ['target'])).to_pandas()
    live = pd.read_parquet(FILE_LIVE)
except Exception as e:
    print(f"   ! DATA LOAD ERROR: {e}")
    sys.exit(1)

available_features = [f for f in feature_names if f in live.columns]
X_train = train[available_features].fillna(0).values
y_train = train['target'].values
X_live = live[available_features].fillna(0).values
live_ids = live.index

print(f"   Training Rows: {len(X_train)} | Live Rows: {len(X_live)}")

# ---------------------------------------------------------
# 3. KERNEL LOGIC (SFL 4-CORE)
# ---------------------------------------------------------
print(">> 3. INITIALIZING SFL KERNELS...")

def make_sig_grav(row):
    total_mass = np.sum(np.abs(row)) + 1e-6
    field_strength = np.mean(np.abs(row)) / total_mass
    ideational_pull = np.sum(row * row) / (total_mass ** 2)
    interpersonal_pull = np.tanh(np.mean(row)) * field_strength
    textual_pull = np.var(row) * field_strength
    field_gravity = np.percentile(np.abs(row), 75) / total_mass
    tenor_gravity = 1 / (1 + np.std(row) + 1e-6)
    mode_gravity = np.mean(np.diff(np.sort(row))) if len(row) > 1 else 0
    return (int(ideational_pull*1000)%5, int(interpersonal_pull*1000)%4, 
            int(textual_pull*1000)%5, int(field_gravity*1000)%3,
            int(tenor_gravity*100)%4, int((mode_gravity+1)*100)%3,
            int(((ideational_pull+interpersonal_pull+textual_pull)/3)*1000)%5)

def make_sig_quantum(row):
    wave_function = np.mean(np.cos(row * np.pi))
    superposition = np.sum(row > np.median(row))
    collapse = np.std(row) / (np.mean(np.abs(row)) + 1e-6)
    spin_up = np.sum(row > 0)
    coherence = 1 / (1 + np.var(row))
    entanglement = 0
    if len(row) > 1:
        entanglement = np.corrcoef([row[:len(row)//2], row[len(row)//2:]])[0, 1]
    return (int(wave_function*1000)%5, int(superposition)%7, 
            int((entanglement+1)*100)%4, int(collapse*100)%5,
            int(spin_up)%6, int(np.sum(row<0))%6, int(coherence*1000)%4)

def make_sig_electro(row):
    electric_field = np.sum(row * np.sign(row))
    magnetic_field = np.sum(np.abs(np.diff(row))) if len(row) > 1 else 0
    photon_energy = np.mean(row ** 2)
    frequency = np.abs(np.fft.fft(row)[1]) if len(row) > 1 else 0
    wavelength = 1 / (np.std(row) + 1e-6)
    polarization = np.mean(row) / (np.std(row) + 1e-6)
    impedance = np.sqrt(np.var(row) / (np.mean(row ** 2) + 1e-6))
    return (int(electric_field*100)%7, int(magnetic_field*100)%6,
            int(photon_energy*1000)%5, int(frequency*100)%4,
            int(wavelength*10)%5, int(polarization*100)%4, int(impedance*1000)%3)

def make_sig_vibe(row):
    amplitude = np.max(np.abs(row))
    frequency = len(np.where(np.diff(np.sign(row)))[0]) if len(row) > 1 else 0
    resonance = np.mean(row) * np.std(row)
    harmonics = np.sum(row[::2]) - np.sum(row[1::2]) if len(row) > 1 else 0
    damping = np.exp(-np.std(row))
    phase = np.angle(np.sum(row * np.exp(1j * np.arange(len(row))))) if len(row) > 0 else 0
    energy = np.sum(row ** 2)
    return (int(amplitude*100)%6, int(frequency)%5, int((resonance+10)*100)%7,
            int(harmonics*100)%4, int(damping*1000)%5, 
            int((phase+np.pi)*100)%4, int(energy*10)%6)

# ---------------------------------------------------------
# 4. CALIBRATION & GENERATION
# ---------------------------------------------------------
print(">> 4. CALIBRATING...")

grav_map, quantum_map, electro_map, vibe_map = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

for row, target in zip(X_train, y_train):
    grav_map[make_sig_grav(row)].append(target)
    quantum_map[make_sig_quantum(row)].append(target)
    electro_map[make_sig_electro(row)].append(target)
    vibe_map[make_sig_vibe(row)].append(target)

grav_final = {k: np.mean(v) for k, v in grav_map.items()}
quantum_final = {k: np.mean(v) for k, v in quantum_map.items()}
electro_final = {k: np.mean(v) for k, v in electro_map.items()}
vibe_final = {k: np.mean(v) for k, v in vibe_map.items()}

print(">> 5. GENERATING & SAVING 6 FILES...")

default_target = np.mean(y_train)

def get_preds(X, lookup, func):
    return np.array([lookup.get(func(r), default_target) for r in X])

def scale(p):
    return (p - p.min()) / (p.max() - p.min() + 1e-9) * 0.6 + 0.2

p_grav = get_preds(X_live, grav_final, make_sig_grav)
p_quant = get_preds(X_live, quantum_final, make_sig_quantum)
p_elec = get_preds(X_live, electro_final, make_sig_electro)
p_vibe = get_preds(X_live, vibe_final, make_sig_vibe)

pd.DataFrame({'id': live_ids, 'prediction': scale(p_grav)}).to_csv(f"gravitational_sfl_{DATE_STR}.csv", index=False)
pd.DataFrame({'id': live_ids, 'prediction': scale(p_quant)}).to_csv(f"quantum_sfl_{DATE_STR}.csv", index=False)
pd.DataFrame({'id': live_ids, 'prediction': scale(p_elec)}).to_csv(f"electromagnetic_sfl_{DATE_STR}.csv", index=False)
pd.DataFrame({'id': live_ids, 'prediction': scale(p_vibe)}).to_csv(f"vibrational_sfl_{DATE_STR}.csv", index=False)

print("   Saved: 4 Substrate Models")

m1 = (p_grav + p_quant + p_elec + p_vibe) / 4.0
pd.DataFrame({'id': live_ids, 'prediction': scale(m1)}).to_csv(f"model1_SFL_NON_NEUTRAL_{DATE_STR}.csv", index=False)
print(f"   Saved: model1_SFL_NON_NEUTRAL_{DATE_STR}.csv")

meta_matrix = np.column_stack([p_grav, p_quant, p_elec, p_vibe])
neutralized = m1.copy()
for i in range(meta_matrix.shape[1]):
    factor = meta_matrix[:, i]
    dot_prod = np.dot(neutralized, factor) / (np.dot(factor, factor) + 1e-9)
    neutralized = neutralized - dot_prod * factor

pd.DataFrame({'id': live_ids, 'prediction': scale(neutralized)}).to_csv(f"model2_SFL_NEUTRALIZED_{DATE_STR}.csv", index=False)
print(f"   Saved: model2_SFL_NEUTRALIZED_{DATE_STR}.csv")

print("=" * 60)
print("SUCCESS.")

import os
import json
import yaml
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import ASSETS_DIR

def processa_esperimenti(cartella_base):
    dati_estratti = []
    
    # os.walk scansiona la cartella principale e tutte le sottocartelle
    for root, dirs, files in os.walk(cartella_base):
        
        # Controlliamo che la cartella contenga entrambi i file necessari
        if 'metrics.json' in files and 'config.yaml' in files:
            percorso_metrics = os.path.join(root, 'metrics.json')
            percorso_config = os.path.join(root, 'config.yaml')
            
            # --- 1. Estrazione dal nome della cartella ---
            nome_cartella = os.path.basename(root)
            dim_match = re.search(r'dim(\d+)', nome_cartella)
            seed_match = re.search(r'seed_(\d+)', nome_cartella)
            ansatz_match = re.search(r'ansatz([a-zA-Z0-9_]+)_key', nome_cartella)
            
            dim = int(dim_match.group(1)) if dim_match else None
            seed = int(seed_match.group(1)) if seed_match else None
            ansatz = ansatz_match.group(1) if ansatz_match else 'Sconosciuto'
            
            # --- 2. Estrazione dal file YAML (per l'Encoding) ---
           # --- 2. Estrazione dal file YAML (per l'Encoding) ---
            with open(percorso_config, 'r') as f_yaml:
                try:
                    config = yaml.safe_load(f_yaml)
                    
                    # Estraiamo 'encoding_function' dal blocco 'experiment_config'
                    encoding = config.get('experiment_config', {}).get('encoding_function', 'Non_Trovato') 
                    if encoding == "rx":
                        encoding = "Ry-Rz"
                            
                    
                except Exception as e:
                    print(f"Errore nella lettura del file YAML in {nome_cartella}: {e}")
                    encoding = 'Errore_Lettura'

            # --- 3. Estrazione dal file JSON (per le Metriche) ---
            with open(percorso_metrics, 'r') as f_json:
                data = json.load(f_json)
                test_data = data.get('test', {})
                report = test_data.get('classification_report', {}) 
                
           # 1. Creiamo le metriche delle classi dinamicamente
            tmp_classi = {}
            metrs = ("recall", "precision", "f1-score")
            
            for i in range(4):
                for metr in metrs:
                    # Rende la prima lettera maiuscola e cambia "-" in "_". Es: "F1_score_0"
                    nome_colonna = f"{upofirst(metr)}_{i}"
                    
                    # Usiamo str(i) perché nel JSON le chiavi delle classi sono stringhe ("0", "1"...)
                    tmp_classi[nome_colonna] = report.get(str(i), {}).get(metr, None)
                    
            # 2. Appendiamo tutto usando l'operatore ** per fondere i dizionari
            dati_estratti.append({
                'Dimensione_Latente': dim,
                'Seed': seed,
                'Ansatz': ansatz,
                'Encoding': encoding,
                'Macro_F1': test_data.get('macro_f1', None),
                'AUC': test_data.get('auc', None),
                'ECE': test_data.get('ece', None),
                **tmp_classi  # <--- MAGIA DI PYTHON: questo operatore riversa tutte le chiavi/valori di tmp_classi qui dentro
            })
            

    dataframe = pd.DataFrame(dati_estratti)
    dataframe.to_csv(ASSETS_DIR / "report_results.csv")
    return dataframe

metrics = {"F1-macro": "Macro_F1", "AUC":"AUC", "ECE":"ECE"}

def create_plot_metrics_aggr(metric: str):
    # --- ESECUZIONE ---
    # Inserisci il percorso della tua cartella principale "experiments"
    CARTELLA_ESPERIMENTI = ASSETS_DIR / "experiments" 

    # 1. Parsing combinato (YAML + JSON)
    df_risultati = processa_esperimenti(CARTELLA_ESPERIMENTI)

    # 2. Aggregazione statistica (Media e Deviazione Standard per i Seed)
    df_aggregato = df_risultati.groupby(['Dimensione_Latente', 'Ansatz', 'Encoding']).agg({
        'Macro_F1': ['mean', 'std'],
        'AUC': ['mean', 'std'],
        'ECE': ['mean', 'std']
    }).reset_index()

    # Pulizia dei nomi delle colonne
    df_aggregato.columns = ['Dimensione', 'Ansatz', 'Encoding', 
                            'F1_Mean', 'F1_Std', 
                            'AUC_Mean', 'AUC_Std', 
                            'ECE_Mean', 'ECE_Std']

    # Salvataggio della Tabella pronta per il report
    print("Scansione completata! Tabella CSV salvata con successo.")

    # 3. Creazione del Grafico
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_risultati, 
        x='Dimensione_Latente', 
        y=metric, 
        hue='Encoding', 
        style='Ansatz',
        markers=True, 
        errorbar='sd' 
    )

    plt.title('Impatto della Compressione Latente (PCA) sulle Performance VQC', fontsize=14)
    plt.xlabel('Dimensione Spazio Latente (D)', fontsize=12)
    plt.ylabel(f'{metric} (Test)', fontsize=12)
    plt.xticks([4, 8, 16, 32]) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    dir = ASSETS_DIR / "plots"
    if not os.path.exists(dir):
        Path(dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(dir / f"Grafico_Ablation_Dimensioni {metric}.png", dpi=300)
    print("Grafico PNG salvato con successo.")

def upofirst(testo):
    if not testo:
        return testo
    return testo[0].upper() + testo[1:]

def create_plot_classes(encoding: str, ansatz: str, metric: str):
    # --- CREAZIONE DEL GRAFICO PER CLASSE ---

    # 1. Filtra i dati per evitare confusione visiva. 
    # Sostituisci 'zzfeaturemap' e 'ttn' con i vincitori effettivi del tuo esperimento!
    CARTELLA_ESPERIMENTI = ASSETS_DIR / "experiments" 

    df_risultati = processa_esperimenti(CARTELLA_ESPERIMENTI)

    df_best = df_risultati[(df_risultati['Encoding'] == encoding) & (df_risultati['Ansatz'] == ansatz)]

    # 2. "Sciogliamo" il dataframe per trasformare le 4 colonne di Recall in una singola variabile
    df_classi = df_best.melt(
        id_vars=['Dimensione_Latente', 'Seed'], # Colonne da mantenere fisse
        value_vars=[f'{upofirst(metric)}_0', f'{upofirst(metric)}_1', f'{upofirst(metric)}_2', f'{upofirst(metric)}_3'], # Colonne da trasformare
        var_name='Classe',
        value_name=f'{upofirst(metric)}'
    )

    # Rinominiamo per una legenda più elegante nel paper (da "Recall_0" a "Classe 0")
    df_classi['Classe'] = df_classi['Classe'].str.replace(f'{upofirst(metric)}_', 'Classe ')

    # 3. Disegniamo il grafico
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_classi, 
        x='Dimensione_Latente', 
        y=f'{upofirst(metric)}', 
        hue='Classe',  # Crea una linea di colore diverso per ogni classe
        palette='Set1', # Un set di colori ben distinguibili
        marker='s', # 's' fa dei quadratini invece dei pallini per variare lo stile
        errorbar='sd' # Calcola in automatico la deviazione standard sui tuoi 3 seed!
    )

    plt.title(f'Andamento della Recall per singola Classe ({upofirst(encoding)} + {upofirst(ansatz)})', fontsize=14)
    plt.xlabel('Dimensione Spazio Latente (PCA)', fontsize=12)
    plt.ylabel(f'{upofirst(metric)} (Test Set)', fontsize=12)
    plt.xticks([4, 8, 16, 32]) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Patologia Retinica', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    dir = ASSETS_DIR / "plots"
    if not os.path.exists(dir):
        Path(dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(dir/ f"Grafico_{upofirst(metric)}_Classi_{upofirst(encoding)}_{upofirst(ansatz)}.png", dpi=300)
    print("Grafico della Recall per classi salvato con successo!")

create_plot_metrics_aggr(metrics["F1-macro"])
#create_plot_classes("Ry-Rz", "ttn", "f1-score")



import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd

class DatasetFromArray(Dataset):
    def __init__(self, images_array, labels_array, transform=None):
        self.images = images_array
        self.labels = labels_array
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long).squeeze()
        return img, label
    


def undersample_dataset(df, label_col='label', random_seed=42):
    """
    Bilancia un dataset riducendo tutte le classi al numero 
    di elementi della classe minoritaria.
    
    Parametri:
    - df: Il DataFrame di pandas originale (con i percorsi immagini e le etichette).
    - label_col: Il nome della colonna che contiene le etichette delle classi.
    - random_seed: Seme per la riproducibilità del campionamento casuale.
    
    Ritorna:
    - Un nuovo DataFrame bilanciato e rimescolato.
    """
    
    # 1. Conta le occorrenze di ogni classe
    class_counts = df[label_col].value_counts()
    
    # 2. Trova la dimensione della classe minoritaria
    min_size = class_counts.min()
    print(f"Dimensione classe minoritaria trovata: {min_size} elementi.")
    
    balanced_dfs = []
    
    # 3. Itera su ogni classe per fare il sottocampionamento
    for label in class_counts.index:
        # Filtra solo i dati di questa specifica classe
        df_class = df[df[label_col] == label]
        
        # Estrai casualmente 'min_size' elementi
        df_class_sampled = df_class.sample(n=min_size, random_state=random_seed)
        
        # Aggiungi alla lista
        balanced_dfs.append(df_class_sampled)
        
    # 4. Unisci tutto e rimescola le righe (frac=1)
    df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return df_balanced
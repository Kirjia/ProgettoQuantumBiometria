from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import skops.io as sio
from pathlib import Path
from utils import SRC_ROOT, ASSETS_DIR

print(f"La root del progetto è: {SRC_ROOT}")

def create_pretrained_pipeline(pretrained_scaler : StandardScaler, pretrained_pca, model):
        return Pipeline([

            ('scaler', FrozenTransformer(pretrained_scaler)),

            ('pca', FrozenTransformer(pretrained_pca)),

            ('model', model)

        ])

class FrozenTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, pretrained_transformer):

        self.pretrained_transformer = pretrained_transformer

       

    def fit(self, X, y=None):

        return self

       

    def transform(self, X): 

        return self.pretrained_transformer.transform(X)
    
    

def get_safe_target_path(user_input_path):
    """
    Unisce la root (src) con l'input dell'utente e verifica che non si esca da src.
    """
    # Rimuoviamo eventuali slash iniziali (es. "/modelli" diventa "modelli") 
    # per evitare che pathlib lo interpreti come un percorso assoluto
    clean_input = user_input_path.lstrip("\\/")
    
    # Uniamo src con il percorso richiesto e risolviamo
    target_path = (SRC_ROOT / clean_input).resolve()
    
    # CONTROLLO DI SICUREZZA: Verifichiamo che il percorso finale sia ANCORA dentro src
    if not target_path.is_relative_to(SRC_ROOT):
        raise PermissionError(f"Accesso negato: il percorso tenta di uscire dalla root di progetto ({SRC_ROOT})")
    
    if not target_path.exists():
        raise FileNotFoundError(f"La cartella {target_path} non esiste all'interno di src.")
        
    return target_path



def load_all_skops(folder_path):
    models = {}
    path = Path(folder_path)
    
    # Cerchiamo tutti i file con estensione .skops
    for file_path in path.glob("*.skops"):
        print(f"Caricamento di: {file_path.name}...")
        
        try:
            # 1. Troviamo i tipi non fidati per questo specifico file
            untrusted_types = sio.get_untrusted_types(file=file_path)
            
            # 2. Carichiamo il file passando la lista trovata
            # Usiamo il nome del file (senza estensione) come chiave del dizionario
            model_name = file_path.stem 
            models[model_name] = sio.load(file_path, trusted=untrusted_types)
            print(f"{model_name} caricato con successo.")
            
        except Exception as e:
            print(f"Errore nel caricamento di {file_path.name}: {e}")
            
    return models
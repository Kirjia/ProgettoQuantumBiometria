# Utils package for ProgettoQuantumBiometria

from pathlib import Path

# 1. Path(__file__) restituisce il percorso di questo __init__.py
# 2. .resolve() calcola il percorso assoluto pulito
# 3. .parent è la cartella 'utils'
# 4. .parent.parent è la cartella 'src'
SRC_ROOT = Path(__file__).resolve().parent.parent

# Opzionale: puoi esporre anche altre cartelle chiave per comodità
ASSETS_DIR = SRC_ROOT / "assets"
MODELS_DIR = SRC_ROOT / "models"

# Opzionale: puoi importare qui i tuoi sottomoduli per rendere 
# l'importazione più pulita nei notebook
from . import dataset
from . import models
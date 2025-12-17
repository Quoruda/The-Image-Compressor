import pickle
import lzma
import os


class CompressedData:
    def __init__(self, author="FractalCompressor", version="1.0"):
        """
        Conteneur flexible pour stocker des données compressées (ou non)
        et leurs métadonnées associées.
        Utilise LZMA (algorithme du format .xz / 7z) pour une compression maximale.
        """
        self.header = {
            "magic": "FRAC_LZMA",  # Signature pour identifier le format
            "version": version,
            "author": author,
            "created_at": None,  # Pourrait être rempli avec time.time()
        }
        self.frames = []  # Liste flexible pour contenir les données de chaque frame

    def add_frame(self, data, shape, is_compressed=True, **metadata):
        """
        Ajoute une frame au conteneur.

        Args:
            data: Les données (liste de transformations si compressé, ou array numpy si brut)
            shape: Tuple (hauteur, largeur) original
            is_compressed: Booléen pour savoir comment traiter la data à la lecture
            **metadata: Arguments nommés libres (ex: block_size=8, quality='high', power=0.5)
        """
        frame_entry = {
            "data": data,
            "shape": shape,
            "is_compressed": is_compressed,
            "meta": metadata  # C'est ici que réside la malléabilité
        }
        self.frames.append(frame_entry)

    def save_to_file(self, filename):
        """
        Enregistre tout le contenu dans un fichier binaire compressé avec LZMA.
        """
        # On structure l'objet complet
        payload = {
            "header": self.header,
            "frames": self.frames
        }

        try:
            # LZMA remplace Gzip. preset=9 assure la compression maximale possible.
            with lzma.open(filename, 'wb', preset=9) as f:
                pickle.dump(payload, f)
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde : {e}")
            return False

    @classmethod
    def load_from_file(cls, filename):
        """
        Charge un fichier .frac (ou autre) et retourne une instance de CompressedData.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Le fichier {filename} n'existe pas.")

        try:
            with lzma.open(filename, 'rb') as f:
                payload = pickle.load(f)

            # Reconstruction de l'objet
            instance = cls()
            instance.header = payload.get("header", {})
            instance.frames = payload.get("frames", [])
            return instance
        except Exception as e:
            print(f"Erreur lors du chargement : {e}")
            return None

    def get_frame(self, index=0):
        """Récupère une frame et ses infos."""
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None

    def __repr__(self):
        return f"<CompressedData frames={len(self.frames)} algo=LZMA version={self.header.get('version')}>"
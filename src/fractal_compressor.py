import numpy as np
from typing import List, Tuple, Optional
import time
from multiprocessing import Pool, cpu_count
from functools import partial


class FractalCompressor:
    def __init__(self, block_size=8, step=8, threshold_mean=0.2, threshold_std=0.2,
                 max_orientations=4, use_parallel=True, n_jobs=None):
        """
        Compresseur fractal avec optimisations de vitesse.

        Args:
            block_size: Taille des blocs cibles (range blocks)
            step: Pas de recherche pour les blocs sources
            threshold_mean: Seuil de filtrage sur la moyenne
            threshold_std: Seuil de filtrage sur l'écart-type
            max_orientations: Nombre max d'orientations testées (1-8)
            use_parallel: Active le traitement parallèle
            n_jobs: Nombre de processus (None = auto)
        """
        self.block_size = block_size
        self.step = step
        self.threshold_mean = threshold_mean
        self.threshold_std = threshold_std
        self.max_orientations = max_orientations
        self.use_parallel = use_parallel
        self.n_jobs = n_jobs or max(1, cpu_count() - 1)

        # Cache pour les transformations
        self._transform_cache = {}

        # Statistiques
        self.last_compression_stats = {}

    def reduce(self, img, factor):
        """Réduit la taille d'une image par moyennage."""
        if factor == 1:
            return img.copy()
        h, w = img.shape
        new_h, new_w = h // factor, w // factor
        return img[:new_h * factor, :new_w * factor].reshape(
            new_h, factor, new_w, factor
        ).mean(axis=(1, 3))

    def get_transforms_fast(self, block, max_count=None):
        """
        Génère seulement les N premières orientations.
        Ordre optimisé : identité, rot180, rot90, rot270, puis flips
        """
        if max_count is None:
            max_count = self.max_orientations

        transforms = []
        if max_count >= 1:
            transforms.append(block)  # Identité (le plus fréquent)
        if max_count >= 2:
            transforms.append(np.rot90(block, 2))  # Rotation 180°
        if max_count >= 3:
            transforms.append(np.rot90(block, 1))  # Rotation 90°
        if max_count >= 4:
            transforms.append(np.rot90(block, 3))  # Rotation 270°
        if max_count >= 5:
            transforms.append(np.flipud(block))  # Flip vertical
        if max_count >= 6:
            transforms.append(np.rot90(np.flipud(block), 1))
        if max_count >= 7:
            transforms.append(np.rot90(np.flipud(block), 2))
        if max_count >= 8:
            transforms.append(np.rot90(np.flipud(block), 3))

        return transforms

    def compute_affine_params_fast(self, source, target):
        """Version optimisée du calcul des paramètres affines."""
        # Précalcul pour éviter les répétitions
        src_flat = source.flatten()
        tgt_flat = target.flatten()
        n = src_flat.size

        # Calcul direct des moindres carrés (plus rapide que lstsq)
        sum_s = np.sum(src_flat)
        sum_t = np.sum(tgt_flat)
        sum_ss = np.sum(src_flat * src_flat)
        sum_st = np.sum(src_flat * tgt_flat)

        denom = n * sum_ss - sum_s * sum_s

        if abs(denom) < 1e-10:
            # Matrice quasi-singulière
            return 0.0, np.mean(tgt_flat), float('inf')

        s = (n * sum_st - sum_s * sum_t) / denom
        o = (sum_t - s * sum_s) / n

        # Clamp pour convergence
        s = np.clip(s, -0.9, 0.9)

        # Erreur
        approx = s * src_flat + o
        error = np.sum((tgt_flat - approx) ** 2)

        return s, o, error

    def _process_target_block(self, args):
        """Traite un bloc cible (pour parallélisation)."""
        y, x, target_block, sources, source_stats = args

        t_mean = np.mean(target_block)
        t_std = np.std(target_block)

        min_error = float('inf')
        best_trans = (0, 0, 0.0, t_mean)

        for src_idx, (source_block, (s_mean, s_std)) in enumerate(zip(sources, source_stats)):
            # Filtrage rapide
            if (abs(s_mean - t_mean) > self.threshold_mean or
                    abs(s_std - t_std) > self.threshold_std):
                continue

            # Test des orientations
            for orientation, candidate in enumerate(
                    self.get_transforms_fast(source_block, self.max_orientations)
            ):
                s, o, error = self.compute_affine_params_fast(candidate, target_block)

                if error < min_error:
                    min_error = error
                    best_trans = (src_idx, orientation, s, o)

        return (y, x, best_trans)

    def compress_frame(self, frame, channel_name="Frame", verbose=True):
        """
        Compresse une frame 2D avec parallélisation.
        """
        start_time = time.time()
        h, w = frame.shape
        source_block_size = self.block_size * 2

        # ========== Extraction des blocs sources ==========
        sources = []
        source_stats = []

        for y in range(0, h - source_block_size + 1, self.step):
            for x in range(0, w - source_block_size + 1, self.step):
                block = frame[y:y + source_block_size, x:x + source_block_size]
                reduced_block = self.reduce(block, 2)
                sources.append(reduced_block)
                source_stats.append((np.mean(reduced_block), np.std(reduced_block)))

        n_sources = len(sources)

        if verbose:
            print(f"  [{channel_name}] Blocs {self.block_size}x{self.block_size}, "
                  f"Step {self.step}, Sources: {n_sources}, "
                  f"Max orientations: {self.max_orientations}")

        # ========== Préparation des tâches ==========
        tasks = []
        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                target_block = frame[y:y + self.block_size, x:x + self.block_size]
                tasks.append((y, x, target_block, sources, source_stats))

        # ========== Traitement parallèle ou séquentiel ==========
        if self.use_parallel and len(tasks) > 100:
            with Pool(processes=self.n_jobs) as pool:
                results = pool.map(self._process_target_block, tasks)
        else:
            results = [self._process_target_block(task) for task in tasks]

        # ========== Reconstruction de l'ordre des transformations ==========
        transform_dict = {(y, x): trans for y, x, trans in results}
        transforms = []
        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                transforms.append(transform_dict.get((y, x), (0, 0, 0.0, 0.0)))

        compression_time = time.time() - start_time

        # ========== Statistiques ==========
        transform_size = len(transforms) * 4 * 4
        original_size = h * w
        compression_ratio = original_size / transform_size if transform_size > 0 else 0

        self.last_compression_stats = {
            'time': compression_time,
            'blocks': len(transforms),
            'compression_ratio': compression_ratio,
            'transform_size': transform_size,
            'original_size': original_size,
            'parallel': self.use_parallel,
            'n_jobs': self.n_jobs if self.use_parallel else 1
        }

        if verbose:
            print(f"  Compression: {compression_time:.2f}s, "
                  f"Ratio: {compression_ratio:.2f}x, "
                  f"Jobs: {self.last_compression_stats['n_jobs']}")

        return transforms

    def decompress_frame(self, transforms, shape, iterations=8, verbose=True):
        """Reconstruit une frame à partir des transformations IFS."""
        start_time = time.time()
        h, w = shape
        factor = 2
        source_block_size = self.block_size * factor

        current_img = np.full((h, w), 0.5, dtype=np.float64)

        # Pré-calcul des coordonnées
        src_coords = [
            (y, x)
            for y in range(0, h - source_block_size + 1, self.step)
            for x in range(0, w - source_block_size + 1, self.step)
        ]

        dest_coords = [
            (y, x)
            for y in range(0, h, self.block_size)
            for x in range(0, w, self.block_size)
        ]

        for iteration in range(iterations):
            new_img = np.zeros((h, w), dtype=np.float64)

            # Extraction des sources
            current_sources = []
            for sy, sx in src_coords:
                block = current_img[sy:sy + source_block_size, sx:sx + source_block_size]
                current_sources.append(self.reduce(block, 2))

            # Application des transformations
            for idx, (dy, dx) in enumerate(dest_coords):
                if idx < len(transforms):
                    src_idx, orientation, s, o = transforms[idx]
                    if src_idx < len(current_sources):
                        src_block = current_sources[src_idx]
                        oriented = self.get_transforms_fast(src_block, 8)[orientation]
                        new_img[dy:dy + self.block_size, dx:dx + self.block_size] = s * oriented + o

            current_img = new_img

        decompression_time = time.time() - start_time

        if verbose:
            print(f"  Décompression: {decompression_time:.2f}s ({iterations} itérations)")

        return np.clip(current_img, 0.0, 1.0)

    def compute_psnr(self, original, reconstructed):
        """Calcule le PSNR entre l'image originale et reconstruite."""
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    def evaluate(self, original, reconstructed):
        """Évalue la qualité de la compression."""
        mse = np.mean((original - reconstructed) ** 2)
        psnr = self.compute_psnr(original, reconstructed)

        stats = self.last_compression_stats.copy()
        stats.update({
            'mse': mse,
            'psnr': psnr,
        })

        return stats

    def print_stats(self, stats):
        """Affiche les statistiques de manière lisible."""
        print("\n=== Statistiques de compression ===")
        print(f"Temps compression:    {stats['time']:.2f}s")
        print(f"Parallélisation:      {stats['parallel']} ({stats['n_jobs']} jobs)")
        print(f"Blocs traités:        {stats['blocks']}")
        print(f"Taille originale:     {stats['original_size']} bytes")
        print(f"Taille compressée:    {stats['transform_size']} bytes")
        print(f"Ratio compression:    {stats['compression_ratio']:.2f}x")
        if 'mse' in stats:
            print(f"MSE:                  {stats['mse']:.6f}")
            print(f"PSNR:                 {stats['psnr']:.2f} dB")
        print("=" * 35)
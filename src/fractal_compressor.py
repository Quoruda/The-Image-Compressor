import numpy as np
import time
from multiprocessing import Pool, cpu_count
from itertools import permutations
import sys


class FractalCompressor:
    def __init__(self, block_size=8, step=8, threshold_mean=0.2, threshold_std=0.2,
                 max_orientations=8, use_parallel=True, n_jobs=None):
        self.block_size = block_size
        self.step = step
        self.threshold_mean = threshold_mean
        self.threshold_std = threshold_std
        self.max_orientations = max_orientations
        self.use_parallel = use_parallel
        self.n_jobs = n_jobs or max(1, cpu_count() - 1)
        self.last_compression_stats = {}

    def reduce(self, img, factor):
        if factor == 1: return img.copy()
        h, w = img.shape
        new_h, new_w = h // factor, w // factor
        return img[:new_h * factor, :new_w * factor].reshape(
            new_h, factor, new_w, factor
        ).mean(axis=(1, 3))

    def get_transforms_fast(self, block, max_count=None):
        if max_count is None: max_count = self.max_orientations
        transforms = []
        if max_count >= 1: transforms.append(block)
        if max_count >= 2: transforms.append(np.rot90(block, 2))
        if max_count >= 3: transforms.append(np.rot90(block, 1))
        if max_count >= 4: transforms.append(np.rot90(block, 3))
        if max_count >= 5: transforms.append(np.flipud(block))
        if max_count >= 6: transforms.append(np.rot90(np.flipud(block), 1))
        if max_count >= 7: transforms.append(np.rot90(np.flipud(block), 2))
        if max_count >= 8: transforms.append(np.rot90(np.flipud(block), 3))
        return transforms

    def compute_affine_params_fast(self, source, target):
        src_flat = source.flatten()
        tgt_flat = target.flatten()
        n = src_flat.size

        sum_s = np.sum(src_flat)
        sum_t = np.sum(tgt_flat)
        sum_ss = np.sum(src_flat * src_flat)
        sum_st = np.sum(src_flat * tgt_flat)

        denom = n * sum_ss - sum_s * sum_s

        if abs(denom) < 1e-10:
            return 0.0, np.mean(tgt_flat), float('inf')

        s = (n * sum_st - sum_s * sum_t) / denom
        o = (sum_t - s * sum_s) / n
        s = np.clip(s, -0.9, 0.9)

        approx = s * src_flat + o
        error = np.sum((tgt_flat - approx) ** 2)
        return s, o, error

    def _classify_block(self, block):
        """
        NOUVEAU : Classification par Quadrants (Jacquin's method).
        Divise le bloc en 4 et retourne l'ordre des indices triés par luminosité.
        Il y a 24 classes possibles (permutations de 0,1,2,3).
        """
        h, w = block.shape
        mh, mw = h // 2, w // 2

        # Calcul de la moyenne des 4 quadrants
        # 0: Haut-Gauche, 1: Haut-Droite, 2: Bas-Gauche, 3: Bas-Droite
        means = [
            np.mean(block[:mh, :mw]),
            np.mean(block[:mh, mw:]),
            np.mean(block[mh:, :mw]),
            np.mean(block[mh:, mw:])
        ]

        # Retourne un tuple d'indices triés par luminosité croissante.
        # Ex: (2, 0, 3, 1) -> Le quadrant 2 est le plus sombre, le 1 le plus clair.
        return tuple(np.argsort(means))

    def _process_target_block(self, args):
        # Note: 'library' est maintenant un dictionnaire (classified_library)
        y, x, target_block, classified_library = args
        t_mean = np.mean(target_block)
        t_std = np.std(target_block)

        # Optimisation 1 : Saut des blocs plats
        if t_std < 0.02:
            return (y, x, (0, 0, 0.0, t_mean))

        # NOUVEAU : On classifie le bloc cible
        target_class = self._classify_block(target_block)

        # On ne récupère QUE les candidats qui ont la même classe
        # Cela réduit l'espace de recherche drastiquement (~ / 24).
        candidates = classified_library.get(target_class, [])

        min_error = float('inf')
        best_trans = (0, 0, 0.0, t_mean)
        acceptable_error = 1e-5

        for source_block, src_idx, orientation, s_mean, s_std in candidates:
            if (abs(s_mean - t_mean) > self.threshold_mean or
                    abs(s_std - t_std) > self.threshold_std):
                continue

            s, o, error = self.compute_affine_params_fast(source_block, target_block)
            if error < min_error:
                min_error = error
                best_trans = (src_idx, orientation, s, o)

                # Optimisation 2 : Sortie anticipée
                if min_error < acceptable_error:
                    break

        return (y, x, best_trans)

    def _execute_parallel_compression(self, tasks, pool, progress_callback):
        results = []
        total_tasks = len(tasks)
        chunksize = max(1, total_tasks // (self.n_jobs * 4))

        iterator = pool.imap_unordered(self._process_target_block, tasks, chunksize=chunksize)

        for i, res in enumerate(iterator):
            results.append(res)
            if progress_callback:
                percent = (i + 1) / total_tasks * 100
                progress_callback(percent)

        return results

    def compress_frame(self, frame, channel_name="Frame", verbose=True, progress_callback=None, pool=None):
        start_time = time.time()
        h, w = frame.shape
        source_block_size = self.block_size * 2

        # 1. Génération et CLASSIFICATION de la bibliothèque
        # On utilise un dictionnaire de listes au lieu d'une liste plate
        library = {p: [] for p in permutations(range(4))}

        raw_source_idx = 0
        count_candidates = 0

        for y in range(0, h - source_block_size + 1, self.step):
            for x in range(0, w - source_block_size + 1, self.step):
                block = frame[y:y + source_block_size, x:x + source_block_size]
                reduced_block = self.reduce(block, 2)

                variations = self.get_transforms_fast(reduced_block, self.max_orientations)

                for orient_idx, transformed_block in enumerate(variations):
                    # On calcule la classe pour CHAQUE orientation
                    # (car tourner le bloc change l'ordre des quadrants)
                    block_class = self._classify_block(transformed_block)

                    s_mean = np.mean(transformed_block)
                    s_std = np.std(transformed_block)

                    # On range dans le bon casier
                    library[block_class].append(
                        (transformed_block, raw_source_idx, orient_idx, s_mean, s_std)
                    )
                    count_candidates += 1

                raw_source_idx += 1

        if verbose:
            print(f"[{channel_name}] Library built. {count_candidates} candidates distributed in 24 classes.")

        # 2. Préparation tâches (on passe le dictionnaire 'library')
        tasks = []
        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                target_block = frame[y:y + self.block_size, x:x + self.block_size]
                tasks.append((y, x, target_block, library))

        # 3. Exécution
        results = []

        if self.use_parallel and len(tasks) > 50:
            if pool:
                results = self._execute_parallel_compression(tasks, pool, progress_callback)
            else:
                with Pool(processes=self.n_jobs) as local_pool:
                    results = self._execute_parallel_compression(tasks, local_pool, progress_callback)
        else:
            total = len(tasks)
            for i, task in enumerate(tasks):
                results.append(self._process_target_block(task))
                if progress_callback:
                    progress_callback((i + 1) / total * 100)

        # 4. Finalisation
        transform_dict = {(y, x): trans for y, x, trans in results}
        transforms = []
        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                transforms.append(transform_dict.get((y, x), (0, 0, 0.0, 0.0)))

        compression_time = time.time() - start_time
        transform_size = len(transforms) * 4 * 4
        original_size = h * w
        ratio = original_size / transform_size if transform_size > 0 else 0

        self.last_compression_stats = {
            'time': compression_time,
            'ratio': ratio,
            'psnr': 0
        }

        if verbose:
            print(f"[{channel_name}] Compressed in {compression_time:.2f}s")

        return transforms

    def decompress_frame(self, transforms, shape, iterations=8, verbose=True):
        # La décompression ne change PAS. Elle utilise juste les index stockés.
        start_time = time.time()
        h, w = shape
        factor = 2
        source_block_size = self.block_size * factor

        current_img = np.full((h, w), 0.5, dtype=np.float64)

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

        for i in range(iterations):
            new_img = np.zeros((h, w), dtype=np.float64)
            current_sources = []
            for sy, sx in src_coords:
                block = current_img[sy:sy + source_block_size, sx:sx + source_block_size]
                current_sources.append(self.reduce(block, 2))

            for idx, (dy, dx) in enumerate(dest_coords):
                if idx < len(transforms):
                    src_idx, orientation, s, o = transforms[idx]
                    if src_idx < len(current_sources):
                        src_block = current_sources[src_idx]
                        oriented = self.get_transforms_fast(src_block, 8)[orientation]
                        new_img[dy:dy + self.block_size, dx:dx + self.block_size] = s * oriented + o
            current_img = new_img

        return np.clip(current_img, 0.0, 1.0)

    def evaluate(self, original, reconstructed):
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        self.last_compression_stats['psnr'] = psnr
        return self.last_compression_stats
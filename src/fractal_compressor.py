import numpy as np
import time
from multiprocessing import Pool, cpu_count
from numba import njit, prange


# --- KERNELS NUMBA ULTRA-RAPIDES ---

@njit(fastmath=True)
def jit_reduce_mean(img, factor):
    """Redimensionne l'image par moyenne (Downsampling)."""
    h, w = img.shape
    new_h = h // factor
    new_w = w // factor
    result = np.zeros((new_h, new_w), dtype=np.float32)

    # Optimisation spécifique pour factor=2 (très fréquent)
    if factor == 2:
        for y in range(new_h):
            for x in range(new_w):
                val = (img[y * 2, x * 2] + img[y * 2 + 1, x * 2] +
                       img[y * 2, x * 2 + 1] + img[y * 2 + 1, x * 2 + 1]) * 0.25
                result[y, x] = val
    else:
        scale = 1.0 / (factor * factor)
        for y in range(new_h):
            for x in range(new_w):
                sum_val = 0.0
                for iy in range(factor):
                    for ix in range(factor):
                        sum_val += img[y * factor + iy, x * factor + ix]
                result[y, x] = sum_val * scale
    return result


@njit(fastmath=True)
def jit_get_8_orientations(block):
    """Génère les 8 isométries d'un bloc carré."""
    # Numba n'aime pas trop la création dynamique de listes d'arrays.
    # On retourne un tenseur 3D (8, H, W)
    h, w = block.shape
    res = np.zeros((8, h, w), dtype=np.float32)

    # 0: Identity
    res[0] = block
    # 1: Rot 180 (Rot90 x 2)
    res[1] = np.rot90(block, 2)
    # 2: Rot 90
    res[2] = np.rot90(block, 1)
    # 3: Rot 270
    res[3] = np.rot90(block, 3)
    # 4: Flip UD
    flip = np.flipud(block)
    res[4] = flip
    # 5: Flip + Rot 90
    res[5] = np.rot90(flip, 1)
    # 6: Flip + Rot 180
    res[6] = np.rot90(flip, 2)
    # 7: Flip + Rot 270
    res[7] = np.rot90(flip, 3)

    return res


@njit(fastmath=True)
def jit_classify_block(block):
    """Retourne un entier unique 0-23 représentant l'ordre des quadrants."""
    h, w = block.shape
    mh, mw = h // 2, w // 2

    m0 = np.sum(block[:mh, :mw])
    m1 = np.sum(block[:mh, mw:])
    m2 = np.sum(block[mh:, :mw])
    m3 = np.sum(block[mh:, mw:])

    # On encode la permutation en un entier simple pour indexer un tableau
    # C'est une astuce pour éviter de gérer des tuples complexes dans Numba
    # ordre: 0=m0, 1=m1, 2=m2, 3=m3
    # On trie manuellement ou via argsort
    vals = np.array([m0, m1, m2, m3])
    idx = np.argsort(vals)

    # Hash unique pour la permutation (base 4) : a*64 + b*16 + c*4 + d
    # Mais plus simple : on utilise l'index direct si on avait une table de lookup.
    # Ici on va juste retourner un tuple simulé par un entier pour le mapping Python
    return idx[0] * 1000 + idx[1] * 100 + idx[2] * 10 + idx[3]


@njit(fastmath=True)
def jit_search_block_in_lib(target_block, lib_pixels, lib_meta, thresh_mean, thresh_std):
    """
    LA BOUCLE CRITIQUE.
    Tout ceci s'exécute en code machine pur, sans Python.

    args:
        target_block: (8, 8) float32
        lib_pixels: (N, 64) float32 [Flattened candidates]
        lib_meta: (N, 4) float32 [src_idx, ori_idx, mean, std]
    """
    best_error = 1e20
    best_res = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # src, ori, s, o

    t_mean = np.mean(target_block)
    t_std = np.std(target_block)

    target_flat = target_block.flatten()
    n = target_flat.size

    # Pré-calculs pour la cible
    sum_t = np.sum(target_flat)

    # Boucle sur TOUS les candidats de la classe (Numba vectorise ça)
    n_candidates = lib_pixels.shape[0]

    for i in range(n_candidates):
        # 1. Filtrage rapide sur moyenne/écart-type
        s_mean = lib_meta[i, 2]
        s_std = lib_meta[i, 3]

        if np.abs(s_mean - t_mean) > thresh_mean: continue
        if np.abs(s_std - t_std) > thresh_std: continue

        # 2. Récupération pixels source
        src_flat = lib_pixels[i]

        # 3. Calcul Affine (Inliné pour performance max)
        sum_s = np.sum(src_flat)
        sum_ss = np.sum(src_flat * src_flat)
        sum_st = np.sum(src_flat * target_flat)  # Le plus lourd : produit scalaire

        denom = n * sum_ss - sum_s * sum_s

        if np.abs(denom) < 1e-9:
            s = 0.0
            o = t_mean
            approx = o
            diff = target_flat - approx
            error = np.sum(diff * diff)
        else:
            s = (n * sum_st - sum_s * sum_t) / denom
            o = (sum_t - s * sum_s) / n

            # Clamp s
            if s > 0.9:
                s = 0.9
            elif s < -0.9:
                s = -0.9

            # Erreur MSE
            # approx = s * src + o
            # diff = tgt - approx
            # error = sum(diff^2)
            # Formule développée plus rapide ? Non, restons sur le vecteur

            error = 0.0
            for k in range(n):
                val = s * src_flat[k] + o
                diff = target_flat[k] - val
                error += diff * diff

        if error < best_error:
            best_error = error
            best_res[0] = lib_meta[i, 0]
            best_res[1] = lib_meta[i, 1]
            best_res[2] = s
            best_res[3] = o

            if best_error < 1e-5:  # Early exit parfait
                break

    return best_res


# --- WORKER FUNCTION (GLOBAL) ---
# Doit être globale pour multiprocessing

def process_chunk_task(args):
    """Traite une bande horizontale de l'image (plusieurs lignes)."""
    (start_y, height_chunk, width, full_img_copy, block_size,
     step, lib_dict_pixels, lib_dict_meta, t_mean_chk, t_std_chk) = args

    results = []

    # On parcourt les blocs de ce chunk
    # Note: On doit s'assurer de ne pas dépasser
    end_y = start_y + height_chunk

    for y in range(start_y, end_y, block_size):
        if y + block_size > full_img_copy.shape[0]: break

        for x in range(0, width, block_size):
            if x + block_size > full_img_copy.shape[1]: break

            target = full_img_copy[y:y + block_size, x:x + block_size]
            t_std = np.std(target)

            # Skip blocs unis
            if t_std < 0.02:
                results.append((y, x, 0, 0, 0.0, np.mean(target)))
                continue

            # Classification
            # On réutilise le petit kernel
            class_key = jit_classify_block(target)

            # Récupération des arrays plats pour cette classe
            # lib_dict_pixels est un dict {int_key: np.array_2d}
            if class_key in lib_dict_pixels:
                lib_px = lib_dict_pixels[class_key]
                lib_mt = lib_dict_meta[class_key]

                # APPEL NUMBA (C'est ici que ça va vite)
                res = jit_search_block_in_lib(target, lib_px, lib_mt, t_mean_chk, t_std_chk)

                # Protection casting: arrondi pour éviter le 199.999 -> 199
                src_idx = int(res[0] + 0.5)
                ori_idx = int(res[1] + 0.5)

                results.append((y, x, src_idx, ori_idx, float(res[2]), float(res[3])))
            else:
                # Fallback si classe vide (rare)
                results.append((y, x, 0, 0, 0.0, np.mean(target)))

    return results


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

    def reduce(self, img, factor):
        return jit_reduce_mean(img.astype(np.float32), factor)

    def get_transforms_fast(self, block, max_count=None):
        # Wrapper pour compatibilité ancien code si besoin, mais on utilise jit en interne
        res = jit_get_8_orientations(block.astype(np.float32))
        return [res[i] for i in range(min(len(res), max_count or 8))]

    def compress_frame(self, frame, channel_name="Frame", verbose=True, progress_callback=None, pool=None):
        start_time = time.time()

        # Assurer float32 pour Numba
        frame = np.ascontiguousarray(frame, dtype=np.float32)
        h, w = frame.shape
        source_block_size = self.block_size * 2

        # --- 1. CONSTRUCTION DE LA BIBLIOTHEQUE (Vectorisée) ---
        # Au lieu de listes de tuples, on veut des Arrays numpy par classe.

        # Structure temporaire : listes Python
        temp_lib = {}  # Key -> list of (flat_pixels, src_idx, ori_idx, mean, std)

        raw_source_idx = 0

        # Note: On pourrait aussi accélérer cette boucle, mais elle est O(N) alors que la recherche est O(N^2)
        # Donc c'est moins grave.
        for y in range(0, h - source_block_size + 1, self.step):
            for x in range(0, w - source_block_size + 1, self.step):
                block = frame[y:y + source_block_size, x:x + source_block_size]
                # Downsample JIT
                reduced = jit_reduce_mean(block, 2)

                # Générer 8 orientations JIT (retourne tenseur 8x8x8)
                orientations = jit_get_8_orientations(reduced)

                for ori_idx in range(self.max_orientations):
                    candidate = orientations[ori_idx]

                    # Classify JIT
                    cls = jit_classify_block(candidate)

                    if cls not in temp_lib: temp_lib[cls] = {'px': [], 'meta': []}

                    c_mean = np.mean(candidate)
                    c_std = np.std(candidate)

                    # On stocke les pixels aplatis
                    temp_lib[cls]['px'].append(candidate.flatten())
                    # On stocke les métadonnées
                    temp_lib[cls]['meta'].append([raw_source_idx, ori_idx, c_mean, c_std])

                raw_source_idx += 1

        # Conversion en NumPy Arrays "Frozens" pour lecture rapide
        lib_dict_pixels = {}
        lib_dict_meta = {}

        total_candidates = 0
        for k in temp_lib:
            # (N, 64)
            arr_px = np.array(temp_lib[k]['px'], dtype=np.float32)
            # (N, 4)
            arr_mt = np.array(temp_lib[k]['meta'], dtype=np.float32)

            lib_dict_pixels[k] = arr_px
            lib_dict_meta[k] = arr_mt
            total_candidates += len(arr_px)

        if verbose:
            print(f"[{channel_name}] Library: {total_candidates} candidates built.")

        # --- 2. PREPARATION DES TACHES (Chunking) ---
        # Au lieu d'une tâche par bloc, on fait une tâche par Core CPU
        # On découpe l'image en bandes horizontales.

        rows_per_job = int(np.ceil(h / self.n_jobs))
        # On s'assure que c'est un multiple de block_size pour pas couper un bloc
        rows_per_job = (rows_per_job // self.block_size) * self.block_size
        if rows_per_job < self.block_size: rows_per_job = self.block_size

        tasks = []
        for y_start in range(0, h, rows_per_job):
            # Arguments pour le worker
            # Attention: On passe des copies ou références read-only.
            # Sur Linux (fork) c'est gratuit. Sur Windows, ça copie (un peu lent mais mieux que 10000 fois).
            task_args = (
                y_start,
                rows_per_job,
                w,
                frame,  # L'image entière (le worker découpera)
                self.block_size,
                self.step,
                lib_dict_pixels,
                lib_dict_meta,
                self.threshold_mean,
                self.threshold_std
            )
            tasks.append(task_args)

        # --- 3. EXECUTION ---
        final_results = []

        if self.use_parallel and pool:
            # imap_unordered permet d'avoir le callback
            total_chunks = len(tasks)
            for i, chunk_res in enumerate(pool.imap_unordered(process_chunk_task, tasks)):
                final_results.extend(chunk_res)
                if progress_callback:
                    progress_callback((i + 1) / total_chunks * 100)
        else:
            # Fallback séquentiel
            for i, t in enumerate(tasks):
                final_results.extend(process_chunk_task(t))
                if progress_callback:
                    progress_callback((i + 1) / len(tasks) * 100)

        # --- 4. RECONSTRUCTION ---
        # Transform map
        transform_map = {}
        for res in final_results:
            y, x, src, ori, s, o = res
            transform_map[(y, x)] = (src, ori, s, o)

        transforms = []
        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                transforms.append(transform_map.get((y, x), (0, 0, 0.0, 0.0)))

        dt = time.time() - start_time
        if verbose:
            print(f"[{channel_name}] Done in {dt:.2f}s.")

        return transforms

    def decompress_frame(self, transforms, shape, iterations=8, verbose=True):
        # La décompression est ultra-légère comparée à la compression,
        # l'optimisation Numba n'est pas critique ici mais on garde la cohérence.
        h, w = shape
        factor = 2
        source_block_size = self.block_size * factor

        # Init gris
        current_img = np.full((h, w), 0.5, dtype=np.float32)

        # Pré-calcul des coordonnées
        dest_coords = []
        idx = 0
        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                dest_coords.append((y, x))

        # Pour récupérer les blocs sources rapidement
        # On ne peut pas facilement vectoriser ça car c'est de l'accès aléatoire
        # Mais 8 itérations sur une image 512x512 c'est < 1 seconde en Python pur.

        for i in range(iterations):
            next_img = np.zeros_like(current_img)

            # Downsample complet de l'image courante
            reduced = jit_reduce_mean(current_img, 2)

            for idx, trans in enumerate(transforms):
                dy, dx = dest_coords[idx]
                src_idx, ori, s, o = trans

                # Retrouver coordonnées source depuis l'index linéaire
                # src_idx correspond au passage (0,0), (step, 0)...
                # nb_blocks_w = (w - src_size) // step + 1

                # Calcul un peu pénible, on aurait dû stocker (sy, sx) ?
                # Non, ça prend trop de place. On recalcule.

                # Refaire le calcul de grille source
                # Largeur grille source
                w_grid = (w - source_block_size) // self.step + 1

                sy_idx = int(src_idx // w_grid)
                sx_idx = int(src_idx % w_grid)

                # FIX CRITIQUE: Ajustement des coordonnées pour l'image réduite (factor 2)
                # Les indices sy_idx * step sont dans l'espace image d'origine.
                # 'reduced' est deux fois plus petite.
                # On doit diviser par 2 pour tomber sur le bon pixel dans 'reduced'.
                sy = (sy_idx * self.step) // 2
                sx = (sx_idx * self.step) // 2

                # Protection
                if sy + self.block_size > reduced.shape[0] or sx + self.block_size > reduced.shape[1]:
                    continue

                # Le bloc réduit est déjà à la taille du bloc destination (grâce au reduce(2))
                # reduced est l'image réduite de moitié.
                # le bloc source original était (2B, 2B) à (sy*2, sx*2)
                # dans l'image reduite, il est à (sy, sx) et fait (B, B)

                src_block = reduced[sy:sy + self.block_size, sx:sx + self.block_size]

                # Appliquer transformation
                # On utilise notre fonction JIT qui retourne les 8 orientations
                # C'est un peu overkill de générer les 8 pour en prendre 1, mais c'est rapide.
                # Ou on fait juste la bonne rotation.

                if src_block.shape != (self.block_size, self.block_size):
                    continue

                # Rotation manuelle rapide (Numpy est bon pour ça)
                if ori == 0:
                    block_rot = src_block
                elif ori == 1:
                    block_rot = np.rot90(src_block, 2)
                elif ori == 2:
                    block_rot = np.rot90(src_block, 1)
                elif ori == 3:
                    block_rot = np.rot90(src_block, 3)
                elif ori == 4:
                    block_rot = np.flipud(src_block)
                elif ori == 5:
                    block_rot = np.rot90(np.flipud(src_block), 1)
                elif ori == 6:
                    block_rot = np.rot90(np.flipud(src_block), 2)
                else:
                    block_rot = np.rot90(np.flipud(src_block), 3)

                # Application affine
                next_img[dy:dy + self.block_size, dx:dx + self.block_size] = s * block_rot + o

            current_img = next_img

        return np.clip(current_img, 0.0, 1.0)
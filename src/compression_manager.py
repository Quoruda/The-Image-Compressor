import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count
from fractal_compressor import FractalCompressor
from compressed_data import CompressedData
from numba import njit


# --- NUMBA OPTIMIZED BIT PACKING ---

@njit(fastmath=True)
def jit_pack_transforms(src_array, ori_array, s_array, o_array, count, packed_dtype_is_uint16):
    """
    Boucle de compactage binaire optimisée Numba.
    Remplace la boucle Python lente.
    """
    # Création du dtype structuré manuellement difficile en Numba pur,
    # on retourne donc des arrays séparés qu'on assemblera ou un array d'entiers si on simplifie.
    # Pour garder la structure 'void' de numpy, on va remplir des arrays simples ici
    # et numpy fera l'assignation finale rapidement.

    packed_src_ori = np.zeros(count, dtype=np.uint32)
    packed_s = np.zeros(count, dtype=np.int8)
    packed_o = np.zeros(count, dtype=np.uint8)

    for i in range(count):
        src = src_array[i]
        ori = ori_array[i]
        s = s_array[i]
        o_val = o_array[i]

        # Bit Packing logic
        packed_src_ori[i] = (src << 3) | (ori & 7)

        # Quantification S
        s_clamped = s * 127.0
        if s_clamped > 127:
            s_clamped = 127
        elif s_clamped < -127:
            s_clamped = -127
        packed_s[i] = int(s_clamped)

        # Quantification O
        o_clamped = (o_val + 1.0) * 85.0
        if o_clamped > 255:
            o_clamped = 255
        elif o_clamped < 0:
            o_clamped = 0
        packed_o[i] = int(o_clamped)

    return packed_src_ori, packed_s, packed_o


class CompressionManager:

    @staticmethod
    def _quantize_transforms(transforms):
        """
        Optimisation Stockage (Bit Packing) avec accélération Numba.
        """
        count = len(transforms)
        if count == 0:
            return np.array([], dtype=np.uint8)

        # Conversion préalable en arrays pour Numba
        # transforms est une liste de tuples (src, ori, s, o)
        # Zip est rapide, mais np.array sur une grosse liste peut prendre un peu de temps.
        # Cependant, le gain sur la boucle de calcul compense.

        # On sépare les composants
        # Note: Cette transformation "zip(*transforms)" est efficace en Python
        srcs, oris, ss, os_vals = zip(*transforms)

        src_arr = np.array(srcs, dtype=np.int32)
        ori_arr = np.array(oris, dtype=np.int32)
        s_arr = np.array(ss, dtype=np.float32)
        o_arr = np.array(os_vals, dtype=np.float32)

        max_src_index = np.max(src_arr)
        max_packed_value = (max_src_index << 3) | 7

        is_uint16 = (max_packed_value < 65536)

        # APPEL JIT
        p_src_ori, p_s, p_o = jit_pack_transforms(src_arr, ori_arr, s_arr, o_arr, count, is_uint16)

        # Construction du tableau structuré final
        if is_uint16:
            packed_dtype = np.uint16
            p_src_ori = p_src_ori.astype(np.uint16)
        else:
            packed_dtype = np.uint32
            # p_src_ori est déjà uint32

        dtype = np.dtype([
            ('src_ori', packed_dtype),
            ('s', np.int8),
            ('o', np.uint8)
        ])

        packed = np.zeros(count, dtype=dtype)
        packed['src_ori'] = p_src_ori
        packed['s'] = p_s
        packed['o'] = p_o

        return packed

    @staticmethod
    def _dequantize_transforms(packed_data):
        """
        Reconstruit les paramètres.
        (La déquantification est rapide, on peut la laisser en NumPy vectorisé pur
         sans forcément utiliser Numba, car NumPy gère le broadcasting très vite).
        """
        if isinstance(packed_data, list):
            return packed_data

        # Version Vectorisée NumPy (plus rapide que la boucle for Python précédente)
        # Pas besoin de Numba ici, NumPy est C-speed sur les arrays entiers.

        raw_vals = packed_data['src_ori'].astype(np.int32)
        srcs = raw_vals >> 3
        oris = raw_vals & 7

        ss = packed_data['s'].astype(np.float32) / 127.0
        os_vals = (packed_data['o'].astype(np.float32) / 85.0) - 1.0

        # Reconstitution de la liste de tuples attendue par le compresseur
        # zip() en python est un itérateur rapide
        return list(zip(srcs, oris, ss, os_vals))

    @staticmethod
    def compress(image: Image.Image, verbose: bool = False, progress_callback=None) -> CompressedData:
        if image.mode != 'YCbCr':
            image = image.convert('YCbCr')

        w, h = image.size
        y_band, cb_band, cr_band = image.split()

        half_w = max(1, w // 2)
        half_h = max(1, h // 2)
        cb_band = cb_band.resize((half_w, half_h), Image.BICUBIC)
        cr_band = cr_band.resize((half_w, half_h), Image.BICUBIC)

        channels_data = [
            ('Y', np.array(y_band) / 255.0),
            ('Cb', np.array(cb_band) / 255.0),
            ('Cr', np.array(cr_band) / 255.0)
        ]

        pixels_y = w * h
        pixels_chroma = half_w * half_h
        total_pixels = pixels_y + pixels_chroma * 2

        weights = {
            'Y': pixels_y / total_pixels * 100,
            'Cb': pixels_chroma / total_pixels * 100,
            'Cr': pixels_chroma / total_pixels * 100
        }

        block_size = 8
        step = 8
        compressor8b = FractalCompressor(block_size=block_size, step=step, max_orientations=8)

        all_transforms = {}
        progress_state = {'base': 0.0}

        def make_channel_callback(weight):
            def callback(percent):
                current_val = progress_state['base'] + (percent / 100.0) * weight
                if progress_callback:
                    progress_callback(current_val)

            return callback

        n_jobs = max(1, cpu_count() - 1)

        try:
            with Pool(processes=n_jobs) as shared_pool:
                for name, img_array in channels_data:
                    if verbose: print(f"Compressing {name}...")

                    cb = make_channel_callback(weights[name])

                    transforms = compressor8b.compress_frame(
                        img_array,
                        channel_name=name,
                        verbose=verbose,
                        pool=shared_pool,
                        progress_callback=cb
                    )

                    packed_transforms = CompressionManager._quantize_transforms(transforms)
                    all_transforms[name] = packed_transforms

                    progress_state['base'] += weights[name]

        except Exception as e:
            print(f"Pool error: {e}")
            raise e

        archive = CompressedData()
        archive.add_frame(
            data=all_transforms,
            shape=(h, w),
            is_compressed=True,
            block_size=block_size,
            step=step,
            original_mode='YCbCr',
            subsampling='4:2:0',
            quantization='int8_packed_numba'
        )

        return archive

    @staticmethod
    def decompress(archive: CompressedData, verbose: bool = False) -> Image.Image:
        frame = archive.get_frame(0)
        if not frame: raise ValueError("Empty archive")
        if not frame['is_compressed']: return Image.fromarray(frame['data'])

        meta = frame['meta']
        data = frame['data']
        shape = frame['shape']

        compressor = FractalCompressor(block_size=meta.get('block_size', 8))
        original_mode = meta.get('original_mode', 'RGB')
        subsampling = meta.get('subsampling', None)

        if original_mode == 'YCbCr' and isinstance(data, dict):
            reconstructed_bands = []
            for channel in ['Y', 'Cb', 'Cr']:
                if channel in data:
                    packed_transforms = data[channel]

                    # Décompression optimisée vectorielle
                    transforms = CompressionManager._dequantize_transforms(packed_transforms)

                    target_h, target_w = shape
                    is_subsampled = (channel in ['Cb', 'Cr'] and subsampling == '4:2:0')

                    if is_subsampled:
                        target_h = max(1, target_h // 2)
                        target_w = max(1, target_w // 2)

                    rec_array = compressor.decompress_frame(transforms, (target_h, target_w), verbose=verbose)
                    rec_img = Image.fromarray((rec_array * 255).astype(np.uint8))

                    if is_subsampled:
                        rec_img = rec_img.resize((shape[1], shape[0]), Image.BICUBIC)
                    reconstructed_bands.append(rec_img)
                else:
                    reconstructed_bands.append(Image.new('L', (shape[1], shape[0]), 128))

            return Image.merge('YCbCr', tuple(reconstructed_bands)).convert('RGB')

        elif original_mode == 'RGB':
            bands = []
            for c in ['R', 'G', 'B']:
                trans = CompressionManager._dequantize_transforms(data[c])
                rec = compressor.decompress_frame(trans, shape, verbose=verbose)
                bands.append(Image.fromarray((rec * 255).astype(np.uint8)))
            return Image.merge('RGB', tuple(bands))

        return Image.fromarray((compressor.decompress_frame(data, shape) * 255).astype(np.uint8))
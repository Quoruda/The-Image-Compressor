import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count
from fractal_compressor import FractalCompressor
from compressed_data import CompressedData


class CompressionManager:

    @staticmethod
    def _quantize_transforms(transforms):
        """
        Optimisation Stockage (Bit Packing) :
        Convertit les données en binaire compact.

        NOUVEAU : Fusionne l'index source (src) et l'orientation (ori)
        dans le même entier pour économiser 1 octet par bloc (20% de gain).
        """
        count = len(transforms)
        if count == 0:
            return np.array([], dtype=np.uint8)

        # 1. Analyse pour la taille optimale
        max_src_index = 0
        for t in transforms:
            if t[0] > max_src_index:
                max_src_index = t[0]

        # On a besoin de stocker (src << 3) | ori.
        # Donc la valeur max stockée sera environ max_src * 8.
        max_packed_value = (max_src_index << 3) | 7

        # Si la valeur combinée tient dans 65535, on reste sur du uint16 (2 octets)
        # Cela nous permet de gérer jusqu'à 8191 blocs sources avec seulement 2 octets !
        if max_packed_value < 65536:
            packed_dtype = np.uint16
        else:
            packed_dtype = np.uint32

        # Structure : src_ori (2/4 octets), s (1 octet), o (1 octet)
        # Total : 4 ou 6 octets par bloc (contre 5 ou 7 avant)
        dtype = np.dtype([
            ('src_ori', packed_dtype),
            ('s', np.int8),
            ('o', np.uint8)
        ])

        packed = np.zeros(count, dtype=dtype)

        for i, (src, ori, s, o) in enumerate(transforms):
            # Bit Packing : On pousse src à gauche et on insère ori dans les 3 bits de droite
            packed[i]['src_ori'] = (src << 3) | (ori & 7)

            # s: Contraste
            packed[i]['s'] = int(np.clip(s * 127, -127, 127))

            # o: Luminosité
            packed[i]['o'] = int(np.clip((o + 1.0) * 85.0, 0, 255))

        return packed

    @staticmethod
    def _dequantize_transforms(packed_data):
        """
        Reconstruit les paramètres en séparant les bits fusionnés.
        """
        if isinstance(packed_data, list):
            return packed_data

        transforms = []
        for i in range(len(packed_data)):
            item = packed_data[i]

            # Unpacking des bits
            val = int(item['src_ori'])
            src = val >> 3  # On récupère l'index (tout sauf les 3 derniers bits)
            ori = val & 7  # On récupère l'orientation (les 3 derniers bits seulement)

            s = float(item['s']) / 127.0
            o = (float(item['o']) / 85.0) - 1.0

            transforms.append((src, ori, s, o))

        return transforms

    @staticmethod
    def compress(image: Image.Image, verbose: bool = False, progress_callback=None) -> CompressedData:
        """
        Gère la compression YCbCr 4:2:0 avec quantification binaire dynamique.
        """
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

        # Poids pour la progress bar
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

                    # Quantification avec Bit Packing
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
            quantization='int8_packed'  # Marqueur mis à jour
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

                    # Décompression avec Unpacking
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

        # Legacy RGB
        elif original_mode == 'RGB':
            bands = []
            for c in ['R', 'G', 'B']:
                trans = CompressionManager._dequantize_transforms(data[c])
                rec = compressor.decompress_frame(trans, shape, verbose=verbose)
                bands.append(Image.fromarray((rec * 255).astype(np.uint8)))
            return Image.merge('RGB', tuple(bands))

        return Image.fromarray((compressor.decompress_frame(data, shape) * 255).astype(np.uint8))
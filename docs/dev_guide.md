# Developer Guide

## Installation

Prerequisites: Python 3.8+

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: On Linux, `python3-tk` must often be installed via apt/dnf/pacman.*

## Using the Application (GUI)

```bash
python src/gui.py
```

## Using the API (Headless)

You can integrate the compressor into your own Python scripts.

### Complete Example

```python
from src.compression_manager import CompressionManager
from PIL import Image

# 1. Load
img = Image.open("holiday_photo.jpg")

# 2. Compress (verbose=True shows progress)
# Returns a CompressedData object
archive = CompressionManager.compress(img, verbose=True)

# 3. Save to disk
archive.save_to_file("souvenir.frac")

# ---

# 4. Load & Decompress
loaded_archive = CompressedData.load_from_file("souvenir.frac")
restored_img = CompressionManager.decompress(loaded_archive)

# 5. Display or Save
restored_img.show()
restored_img.save("restored.png")
```

## Tests and Experimentation

A `tests.ipynb` Jupyter notebook is available at the root. It allows you to:
- Visualize source and target blocks.
- Test the impact of block size (4, 8, 16) on quality and size.
- Analyze Numba vs pure Python performance.

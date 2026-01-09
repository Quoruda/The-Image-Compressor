# The Fractal Compressor

An experimental image compression application based on the theory of Partitioned Iterated Function Systems (PIFS) (Fractal Compression).

This project explores how images can be encoded not as a grid of pixels, but as a set of mathematical equations that reference themselves.

## Features

- **Fractal Compression**: Uses the image's internal redundancy (self-similarity) to compress it.
- **Performance**: Compression engine accelerated via Numba (JIT) and Multiprocessing.
- **.frac Format**: Custom file format with LZMA container for maximum density.
- **GUI**: Integrated and easy-to-use viewer (based on ttkbootstrap).
- **Optimizations**: Support for YCbCr 4:2:0 chroma subsampling.

## Installation

Make sure you have Python 3.8+ installed.

### 1. System Dependencies (Tkinter)

This application uses Tkinter for its interface. While part of the standard Python library, it is often packaged separately on Linux and might be missing from minimal installations or virtual environments.

If you encounter an error like `ModuleNotFoundError: No module named 'tkinter'`, please install it manually:

**Debian / Ubuntu / Mint:**
```bash
sudo apt-get update
sudo apt-get install python3-tk
```

**Fedora / RHEL:**
```bash
sudo dnf install python3-tkinter
```

**Arch Linux:**
```bash
sudo pacman -S tk
```

**Windows / macOS:** Usually included with the official Python installer. Ensure the "tcl/tk and IDLE" option was checked during installation.

### 2. Python Dependencies

Clone or download this repository.

Install the required libraries via requirements.txt:

```bash
pip install -r requirements.txt
```

**Note:** Main dependencies include numpy, Pillow, ttkbootstrap, and numba.

## Usage

To launch the graphical application:

```bash
python gui.py
```

1. Click on **Open** to load an image (JPG, PNG, BMP...).
2. Click on **Compress** to start processing.
3. Once finished, you can save the result via **Save** (.frac format).

## Documentation

For in-depth technical details on the algorithm, quadrant classification, or vectorized implementation, please consult the `/Doc` folder.

## Warning

This is a research/demonstration project. Although functional, compression time can be significant on high-resolution images (4K) due to the mathematical complexity of fractals (O(n^2)).
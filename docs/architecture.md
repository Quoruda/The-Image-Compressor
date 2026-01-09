# Technical Architecture

The project is divided into four main components located in the `src/` folder. This separation isolates the complex mathematical logic from the user interface.

## 1. Compression Engine
**File:** `src/fractal_compressor.py`

This is the core of the system. It contains the mathematical intelligence and performance optimizations.
- **Role**: Analyzes the image to find fractal matches.
- **Optimization**: Heavily uses `numba` (`@njit`) to compile critical loops into machine code on the fly.
- **Parallelization**: Splits the image into horizontal strips to distribute the load across all CPU cores via `multiprocessing`.

## 2. Flow Manager
**File:** `src/compression_manager.py`

This module acts as the conductor.
- **Role**: Prepares the image before compression (YCbCr conversion, 4:2:0 resizing for chrominance).
- **Quantization**: Handles the conversion of floating-point numbers (engine results) into compact integers (Bit Packing) before storage.

## 3. Data Manager
**File:** `src/compressed_data.py`

Responsible for data persistence.
- **Role**: Defines the structure of the `.frac` archive.
- **Final Compression**: Applies LZMA compression (7-zip algorithm) on the serialized data to gain an additional 20-30% space.

## 4. Graphical Interface
**File:** `src/gui.py`

The user interface for interacting with the compressor.
- **Tech Stack**: `tkinter` with the `ttkbootstrap` (Cosmo) theme.
- **Features**: Real-time visualization, threading to avoid blocking the UI during compression, and native image display management.

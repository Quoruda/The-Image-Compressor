# File Format (.frac)

The `.frac` format is a custom binary format designed for maximum information density.

## File Structure

The file consists of two main parts:
1.  **Header**: A dictionary containing the version, author, and `FRAC_LZMA` signature.
2.  **Frame Payload**: A list of "frames", potentially supporting video in the future.

Everything is serialized with `pickle` and then compressed using **LZMA** (preset=9).

## Bit Packing & Quantization

Before LZMA compression, mathematical data (which are `floats`) are quantized to fit into a minimum number of bits. This is lossy compression but imperceptible if well-calibrated.

### Encoding a Transformation
A fractal transformation for a block is stored in **4 or 6 bytes** maximum (vs ~16 bytes minimum for raw floats).

| Data | Original Type | Storage | Method |
| :--- | :--- | :--- | :--- |
| **Source Index + Orientation** | `int` + `int` | `uint16` or `uint32` | The 3 LSBs encode orientation (0-7), the rest codes the source block index. |
| **Scale ($s$)** | `float` | `int8` (1 byte) | Mapped to [-127, 127] (representing [-1.0, 1.0]). |
| **Offset ($o$)** | `float` | `uint8` (1 byte) | Mapped to [0, 255]. |

This "Bit Packing" step reduces the raw data size by a factor of 2 to 4 even before the final LZMA compression step.

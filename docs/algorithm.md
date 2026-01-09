# Compression Algorithm (PIFS)

The central idea of fractal compression (Partitioned Iterated Function Systems) is that "the whole resembles its parts". The image is not stored pixel by pixel, but as a series of transformations indicating how to reconstruct the image from itself.

## The Concept

For each block of the target image (**Range Block** - $R$), the algorithm searches for a block in the source image (**Domain Block** - $D$, usually a reduced version of the image) that most closely resembles it.

The sought affine transformation equation is:
$$ R \approx s \cdot \text{Iso}(D) + o $$

Where:
- **$s$ (Scale)**: Contrast factor. Adjusts the block's dynamic range.
- **$o$ (Offset)**: Average brightness. Adjusts the global gray level.
- **$\text{Iso}$**: A geometric transformation (Isometry).

## Isometries
To maximize the chances of finding a match, the source block can be geometrically manipulated. We test 8 variants:
1. Identity
2. Rotations (90°, 180°, 270°)
3. Mirror (Flip)
4. Mirror + Rotations

## Critical Optimizations

An exhaustive search (comparing every target block with every possible source block) would be too slow ($O(N^2)$).

### Block Classification
Each $8\times8$ block is analyzed and classified according to its brightness distribution across its 4 quadrants (top-left, top-right, bottom-left, bottom-right).
- There are $4! = 24$ possible brightness order permutations.
- **Rule**: We only search for a match among source blocks having the **same class** (same brightness signature).
- This reduces the search space by a factor of ~24.

### Numba and JIT Acceleration
Heavy mathematical functions (mean, standard deviation, dot product for linear regression to find $s$ and $o$) are compiled into native code via Numba, making Python as fast as C for these sections.

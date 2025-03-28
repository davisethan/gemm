# General Matrix Multiplication (GEMM)

## Installation

```bash
# Download
wget https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.29.tar.gz
tar -xvzf v0.3.29.tar.gz
cd OpenBLAS-0.3.29

# Install
make -j$(nproc) USE_OPENMP=1
make PREFIX=~/openblas install
```

## Execute All Tests

```bash
./execute.sh
```

## Single Compilation & Execution

```bash
./compile.sh <matrix-size>
./main
./mainCuda
```

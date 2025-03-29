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

## Compile & Execute One Test

```bash
./compile.sh <matrix-size>
./main
./mainCuda
```

## Compile & Execute All Tests

```bash
./execute.sh
```

### Sample Results

![Performance v.s. Matrix Size](plot.png)

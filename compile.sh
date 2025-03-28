#!/bin/bash
CPPFLAGS="-I$HOME/openblas/include"
LDFLAGS="-L$HOME/openblas/lib"
g++ -DSIZE=$1 utility.cpp main.cpp -o main $LDFLAGS $CPPFLAGS -fopenmp -lopenblas
nvcc -DSIZE=$1 utilityCuda.cu mainCuda.cu -o mainCuda -lcublas

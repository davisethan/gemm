#!/bin/bash
sizes=(100 200 300 400 500 600 700 800 900
    1000 2000 3000 4000 5000 6000 7000 8000 9000
    10000)
for size in "${sizes[@]}"; do
    ./compile.sh $size
    ./mainCuda
    echo ""
    ./main
    echo ""
done

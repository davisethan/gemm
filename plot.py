import matplotlib.pyplot as plt

def flops(x):
    return 2 * x ** 3 - x ** 2

def sec(x):
    return x * 1000

x = [100, 200, 300, 400, 500, 600, 700, 800, 900,\
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,\
    10000]

yCuBlasExecution = [12.1239, 12.3919, 12.7114, 13.5087, 13.4809, 14.0803, 16.5676, 16.8032, 18.5497,\
                    20.428, 77.6056, 201.553, 460.183, 844.022, 1404.18, 2237.99, 3277.26, 4674.69,\
                        6421.63]
yCuBlasCopy = [0.05392, 0.139782, 0.283787, 0.604936, 0.879311, 1.27614, 1.50936, 1.98802, 2.25274,\
               2.81804, 8.51289, 18.6045, 32.2203, 49.5934, 71.0513, 96.5657, 124.469, 158.923,\
                196.069]
yCuBlasExecutionPerformance = [flops(a) / sec(b) for a, b in zip(x, yCuBlasExecution)]

yCudaExecution = [14.0423, 12.8468, 12.1996, 13.8598, 14.5077, 16.727, 15.0792, 75.5817, 67.5975,\
                  20.334, 68.4376, 174.133, 383.592, 750.296, 1326.41, 2062.2, 3032.66, 4343.42,\
                    5921.06]
yCudaCopy = [0.01944, 0.061248, 0.107456, 0.276525, 0.279955, 0.389237, 0.569559, 0.65753, 0.766972,\
             0.894693, 2.67399, 6.23461, 10.284, 16.3673, 22.7102, 31.0581, 40.2356, 50.8241,\
                62.3618]
yCudaExecutionPerformance = [flops(a) / sec(b) for a, b in zip(x, yCudaExecution)]

yBlas = [0.177878, 0.440823, 1.40575, 2.12585, 3.84999, 6.8264, 10.605, 15.6628, 21.5837,\
         29.1044, 219.05, 734.403, 1728.98, 3374.95, 5821.96, 9292.43, 13842.6, 19686.7,\
            26908.3]
yBlasPerformance = [flops(a) / sec(b) for a, b in zip(x, yBlas)]

yOmp = [11.5182, 3.36564, 9.37084, 28.6293, 31.9821, 50.4266, 81.6964, 122.81, 167.528,\
        256.825, 1752.29, 5893.96, 14070, 33083.9, 56992.4, 74486.6, 135307, 192382,\
            215900]
yOmpPerformance = [flops(a) / sec(b) for a, b in zip(x, yOmp)]

yCppThreads = [4.36398, 10.9308, 14.9365, 24.1694, 38.3974, 60.491, 87.492, 127.007, 175.448,\
               234.576, 1807.05, 6089.15, 14544.1, 28168.2, 48578.1, 77554.2, 116749, 164511,\
                224951]
yCppThreadsPerformance = [flops(a) / sec(b) for a, b in zip(x, yCppThreads)]

plt.figure()
plt.plot(x, yCuBlasExecutionPerformance, label="CuBLAS")
plt.plot(x, yCudaExecutionPerformance, label="CUDA")
plt.plot(x, yBlasPerformance, label="BLAS")
plt.plot(x, yOmpPerformance, label="OpenMP")
plt.plot(x, yCppThreadsPerformance, label="C++ Threads")
plt.xlabel("Matrix Size (N)")
plt.ylabel("Performance (FLOPS)")
plt.title("Performance v.s. Matrix Size")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.loglog(x, yCuBlasExecutionPerformance, label="CuBLAS")
plt.loglog(x, yCudaExecutionPerformance, label="CUDA")
plt.loglog(x, yBlasPerformance, label="BLAS")
plt.loglog(x, yOmpPerformance, label="OpenMP")
plt.loglog(x, yCppThreadsPerformance, label="C++ Threads")
plt.xlabel("Matrix Size (N)")
plt.ylabel("Performance (FLOPS)")
plt.title("Performance v.s. Matrix Size")
plt.legend()
plt.tight_layout()
plt.show()

CC = gcc
MPICC = mpicc
CFLAGS = -Wall -O3
OMP_FLAG = -fopenmp
NUMA_LIBS = -lnuma -lpthread

# 默认编译所有目标
all: serial mpi openmp_static openmp_dynamic openmp_NUMA mpi_NUMA

# 串行版本
serial: serial.c Sort.o
	$(CC) $(CFLAGS) $^ -o $@

# 基础MPI版本
mpi: mpi.c Sort.o
	$(MPICC) $(CFLAGS) $^ -o $@

# 基础OpenMP版本
openmp_static: openmp_static.c Sort.o
	$(CC) $(CFLAGS) $^ -o $@ $(OMP_FLAG)

# 服务器版本（假设需要OpenMP）
openmp_dynamic: openmp_dynamic.o Sort.o
	$(CC) $(CFLAGS) $^ -o $@ $(OMP_FLAG)

# NUMA优化的OpenMP版本
openmp_NUMA: openmp_NUMA.o Sort.o
	$(CC) $(CFLAGS) $^ -o $@ $(OMP_FLAG) $(NUMA_LIBS)

# NUMA优化的MPI版本
mpi_NUMA: mpi_NUMA.o Sort.o
	$(MPICC) $(CFLAGS) $^ -o $@ $(OMP_FLAG) $(NUMA_LIBS)

# 对象文件编译规则
Sort.o: Sort.c Sort.h
	$(CC) $(CFLAGS) -c $< -o $@

openmp_NUMA.o: openmp_NUMA.c
	$(CC) $(CFLAGS) -c $< -o $@ $(OMP_FLAG) $(NUMA_LIBS)

mpi_NUMA.o: mpi_NUMA.c
	$(MPICC) $(CFLAGS) -c $< -o $@ $(OMP_FLAG) $(NUMA_LIBS)

openmp_dynamic.o: openmp_dynamic.c
	$(CC) $(CFLAGS) -c $< -o $@ $(OMP_FLAG)

# 清理
clean:
	rm -f *.o serial mpi openmp_static openmp_dynamic openmp_NUMA mpi_NUMA
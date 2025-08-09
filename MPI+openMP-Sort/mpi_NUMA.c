#include "Sort.h"
#include <mpi.h>
#define __USE_GNU
#include <numa.h>
#include <omp.h>
#include <sched.h>
#include <pthread.h>
int rank, size, n = ELEMENTS;
// NUMA节点绑定（双路NUMA：rank偶数绑定Node0，奇数绑定Node1）
void bind_to_numa_node(int rank) {
    if (numa_available() == -1) return;
    int numa_node = rank % 2;
    numa_run_on_node(numa_node);
    numa_set_preferred(numa_node);
}

// 线程绑定到物理核心（每NUMA节点14物理核）
void bind_threads() {
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int core_id = thread_id + 14 * (numa_node_of_cpu(sched_getcpu()) % 2);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    }
}

void paraQuickSort(int* data, int start, int end, int m, int id, int nowID, int N) {
    int r = end, length = -1;
    int* t = NULL;
    MPI_Status status;

    /* 终止条件：使用OpenMP加速本地排序 */
    if (m == 0) {
        if (nowID == id) {
            #pragma omp parallel num_threads(14)  // 每NUMA节点14线程
            {
                bind_threads();
                QuickSort(data, start, end);
            }
        }
        return;
    }

    /* NUMA感知数据分发 */
    if (nowID == id) {
        while (id + (1 << (m-1)) > N && m > 0) m--;
        if (id + (1 << (m-1)) < N) {
            r = Partition(data, start, end);
            length = end - r;
            MPI_Send(&length, 1, MPI_INT, id + (1 << (m-1)), nowID, MPI_COMM_WORLD);
            if (length > 0) {
                // 优先发送到同NUMA节点的进程
                int target_rank = id + (1 << (m-1));
                if (target_rank % 2 == rank %2) {  // 同NUMA节点
                    MPI_Send(data + r + 1, length, MPI_INT, target_rank, nowID, MPI_COMM_WORLD);
                } else {  // 跨NUMA节点，使用缓冲发送减少延迟
                    int* buf = numa_alloc_local(length * sizeof(int));
                    memcpy(buf, data + r + 1, length * sizeof(int));
                    MPI_Send(buf, length, MPI_INT, target_rank, nowID, MPI_COMM_WORLD);
                    numa_free(buf, length * sizeof(int));
                }
            }
        }
    }

    /* NUMA本地数据接收 */
    if (nowID == id + (1 << (m-1))) {
        MPI_Recv(&length, 1, MPI_INT, id, id, MPI_COMM_WORLD, &status);
        if (length > 0) {
            t = numa_alloc_local(length * sizeof(int));  // 本地NUMA分配
            MPI_Recv(t, length, MPI_INT, id, id, MPI_COMM_WORLD, &status);
        }
    }

    /* 递归排序 */
    int j = r - 1 - start;
    MPI_Bcast(&j, 1, MPI_INT, id, MPI_COMM_WORLD);
    if (j > 0) paraQuickSort(data, start, r - 1, m - 1, id, nowID, N);

    j = length;
    MPI_Bcast(&j, 1, MPI_INT, id, MPI_COMM_WORLD);
    if (j > 0) paraQuickSort(t, 0, length - 1, m - 1, id + (1 << (m-1)), nowID, N);

    /* 结果归并优化 */
    if ((nowID == id + (1 << (m-1))) && (length > 0)) {
        MPI_Send(t, length, MPI_INT, id, id + (1 << (m-1)), MPI_COMM_WORLD);
        numa_free(t, length * sizeof(int));
    }

    if ((nowID == id) && (id + (1 << (m-1)) < N )&& (length > 0)) {
        MPI_Recv(data + r + 1, length, MPI_INT, id + (1 << (m-1)), id + (1 << (m-1)), MPI_COMM_WORLD, &status);
    }
}

int main(int argc, char* argv[]) {
    int* data = NULL;
    
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* NUMA初始化与绑定 */
    bind_to_numa_node(rank);

    /* 根进程初始化数据（交错分配） */
    if (rank == 0) {
        data = numa_alloc_interleaved(n * sizeof(int));
        read_random_array(data, "random_array.bin");
    }

    /* 计算进程拓扑 */
    int m = 0;
    while ((1 << m) <= size) m++;
    m--;

    /* 执行排序 */
    start_time = MPI_Wtime();
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    paraQuickSort(data, 0, n - 1, m, 0, rank, size);
    end_time = MPI_Wtime();

    /* 结果输出 */
    if (rank == 0) {
        printf("前10个元素: ");
        for (int i = 0; i < 10 && i < n; i++) printf("%d ", data[i]);
        printf("\nNUMA优化MPI+OpenMP耗时: %.3fs (进程数=%d)\n", end_time - start_time, size);
        numa_free(data, n * sizeof(int));
    }

    MPI_Finalize();
    return 0;
}
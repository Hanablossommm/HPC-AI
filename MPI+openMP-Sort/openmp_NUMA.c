#include "Sort.h"
#include <omp.h>
#define __USE_GNU
#include <numa.h>
#include <sched.h>
#include <pthread.h>

// 线程绑定到物理核心
void bind_thread_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
}

// NUMA优化的并行快速排序
void quickSort_numa(int* data, int start, int end, int numa_node) {
    if (start < end) {
        int pos = Partition(data, start, end);
        
        #pragma omp task firstprivate(data, start, pos, numa_node)
        {
            numa_run_on_node(numa_node);  // 绑定到指定NUMA节点
            quickSort_numa(data, start, pos - 1, numa_node);
        }
        
        #pragma omp task firstprivate(data, pos, end, numa_node)
        {
            numa_run_on_node(numa_node);  // 绑定到指定NUMA节点
            quickSort_numa(data, pos + 1, end, numa_node);
        }
    }
}

int main(int argc, char* argv[]) {
    // NUMA初始化
    if (numa_available() == -1) {
        fprintf(stderr, "NUMA not available!\n");
        return 1;
    }

    int threads_per_numa = atoi(argv[2]);
    int total_threads = 2 * threads_per_numa;  // 双路NUMA
    int* array = numa_alloc_interleaved(ELEMENTS * sizeof(int));  // 交错分配内存

    read_random_array(array,"random_array.bin");

    double start = omp_get_wtime();
    omp_set_num_threads(total_threads);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int numa_node = thread_id / threads_per_numa;  // 0或1
        bind_thread_to_core(thread_id + numa_node * 28);  // 假设每NUMA节点28逻辑核心
        
        #pragma omp single
        {
            // 为每个NUMA节点创建任务组
            #pragma omp taskgroup
            {
                #pragma omp task
                quickSort_numa(array, 0, ELEMENTS/2 - 1, 0);  // NUMA节点0处理前半
                
                #pragma omp task
                quickSort_numa(array, ELEMENTS/2, ELEMENTS-1, 1);  // NUMA节点1处理后半
            }
        }
    }
    
    double end = omp_get_wtime();
    printf("NUMA优化耗时: %.3f秒 (%d线程)\n", end - start, total_threads);

    numa_free(array, ELEMENTS * sizeof(int));
    return 0;
}
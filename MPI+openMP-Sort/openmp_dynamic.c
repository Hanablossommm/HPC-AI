#include "Sort.h"
#include <omp.h>
/*
task 会将每次递归调用转化为 任务，放入任务池。

所有线程（n 个）从任务池 动态获取任务，避免线程爆炸。
*/
void quickSort2(int* data, int start, int end)  //并行快排
{
    if (start < end) {
        int pos = Partition(data, start, end);
        
        
            #pragma omp task//每次递归调用quick_sort时，生成一个独立的任务（task），由OpenMP运行时动态分配给空闲线程执行。
            quickSort2(data, start, pos - 1);
            #pragma omp task//任务池：所有线程从共享的任务池中获取任务执行，实现动态负载均衡。
            quickSort2(data, pos + 1, end);
        
    }
}


int main(int argc, char* argv[]) {
    int *array = malloc(ELEMENTS* sizeof(int));
    read_random_array(array, "random_array.bin");
    double start = omp_get_wtime();
    int n_threads = atoi(argv[2]);  // 从命令行读取
    omp_set_num_threads(n_threads);
    #pragma omp parallel//创建一个并行区域，默认生成与CPU核心数相等的线程（例如在56线程的服务器上会生成56个线程）。
    {
        #pragma omp single//指定只有一个线程（通常是主线程）执行quick_sort的初始调用，其他线程等待任务分配。
        quickSort2(array, 0, ELEMENTS- 1);
    }
    double end = omp_get_wtime();
    printf("动态线程耗时: %.3f秒\n", end - start);
    free(array);
    return 0;
}

/*
单线程快速排序	多线程快速排序（OpenMP）
时间复杂度：O(n log n)	时间复杂度：O(n log n / p)，其中p为线程数
*/

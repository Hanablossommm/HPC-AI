#include"Sort.h"
#include<omp.h>
/*
每次进入 parallel sections 时，OpenMP 会创建 2 个线程（每个 section 一个）。

如果递归深度很深，会 指数级增加线程数（2^递归深度），但实际受限于 omp_set_num_threads(n) 设置的最大线程数。

如果 n=4，但递归深度为 3，理论上需要 8 个线程，但实际只有 4 个线程可用，部分任务会 串行执行。

线程可能 闲置，因为 sections 是静态划分，无法动态调度。
 */
void quickSort(int* data, int start, int end)  //并行快排
{
    if (start < end) {
        int pos = Partition(data, start, end);
        #pragma omp parallel sections    //设置并行区域
        {
            #pragma omp section          //该区域对前部分数据进行排序
            quickSort(data, start, pos - 1);
            #pragma omp section          //该区域对后部分数据进行排序
            quickSort(data, pos + 1, end);
        }
    }
}

int main(int argc, char* argv[])
{
    int n = atoi(argv[2]), i;   //线程数
    int size = ELEMENTS;   //数据大小
    int* num = (int*)malloc(sizeof(int) * size);
    read_random_array(num,"random_array.bin");

    double starttime = omp_get_wtime();
    omp_set_num_threads(n);   //设置线程数
    quickSort(num, 0, size - 1);   //并行快排
    double endtime = omp_get_wtime();

    for (i = 0; i < 10 && i<size; i++)//输出前十个元素
        printf("%d ", num[i]);
    printf("\n静态线程时间：%lfs\n", endtime - starttime);
    return 0;
}

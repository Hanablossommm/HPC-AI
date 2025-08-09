#include "Sort.h"
int main() {
    int* array = malloc(ELEMENTS * sizeof(int));
    read_random_array(array, "random_array.bin");
    clock_t start = clock();
    QuickSort(array, 0, ELEMENTS - 1);
    clock_t end = clock();
    printf("单机单线程耗时: %.3f秒\n", (double)(end - start) / CLOCKS_PER_SEC);
    
    free(array);
    return 0;
}

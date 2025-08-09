#include  "Sort.h"
void generate_random_array(const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file for writing");
        exit(EXIT_FAILURE);
    }
    
    srand(time(NULL));  // 初始化随机数种子
    
    for (int i = 0; i < ELEMENTS; i++) {
        int num = rand();
        fwrite(&num, sizeof(int), 1, file);
    }
    
    fclose(file);
    printf("Generated random array file: %s with %d integers\n", filename, ELEMENTS);
}

void read_random_array(int * array,const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file for reading");
        exit(EXIT_FAILURE);
    }
    
    
    if (!array) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }
    
    size_t read_count = fread(array, sizeof(int), ELEMENTS, file);
    if (read_count != ELEMENTS) {
        fprintf(stderr, "Only read %zu of %d elements\n", read_count, ELEMENTS);
    }
    
    // 示例：打印前10个元素
    printf("First 10 elements:\n");
    for (int i = 0; i < 10 && i < ELEMENTS; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
    
    fclose(file);
}
//求2的n次方
int exp_2(int n)
{
    int i = 1;
    while (n-- > 0) i *= 2;
    return i;
}

//求以2为底n的对数，向下取整
int log_2(int n)
{
    int i = 1, j = 2;
    while (j < n) {
        j *= 2;
        i++;
    }
    return i;
}
void QuickSort(int* data, int start, int end)  //串行快排
{
    if (start < end) {    //未划分完
        int r = Partition(data, start, end);   //继续划分，进行递归排序
        QuickSort(data, start, r - 1);
        QuickSort(data, r + 1, end);
    }
}

int Partition(int* data, int start, int end)   //划分数据
{
    int temp = data[start];   //以第一个元素为基准
    while (start < end) {
        while (start < end && data[end] >= temp)end--;   //找到第一个比基准小的数
        data[start] = data[end];
        while (start < end && data[start] <= temp)start++;    //找到第一个比基准大的数
        data[end] = data[start];
    }
    data[start] = temp;   //以基准作为分界线
    return start;
}
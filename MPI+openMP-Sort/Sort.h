// File: Sort.h
#ifndef SORT_H  // 头文件保护
#define SORT_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define ELEMENTS 10000000

void read_random_array(int *array, const char* filename);
int exp_2(int n);
int log_2(int n);
void QuickSort(int* data, int start, int end) ; //串行快排
int Partition(int* data, int start, int end) ;  //划分数据
#endif //
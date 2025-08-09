#include"Sort.h"
#include<mpi.h>


void paraQuickSort(int* data, int start, int end, int m, int id, int nowID, int N)
{
    int i, j, r = end, length = -1;  //r表示划分后数据前部分的末元素下标，length表示后部分数据的长度
    int* t;
    MPI_Status status;
    /*终止条件检查*/
    if (m == 0) {   //无进程可以调用
        if (nowID == id) QuickSort(data, start, end);
        return;
    }
    /*数据分发（主进程）*/
    if (nowID == id) {    //当前进程是负责分发的
        while (id + exp_2(m - 1) > N && m > 0) m--;   //寻找未分配数据的可用进程
        if (id + exp_2(m - 1) < N) {  //还有未接收数据的进程，则划分数据
            r = Partition(data, start, end);
            length = end - r;
            MPI_Send(&length, 1, MPI_INT, id + exp_2(m - 1), nowID, MPI_COMM_WORLD);
            if (length > 0)   //id进程将后部分数据发送给id+2^(m-1)进程
                MPI_Send(data + r + 1, length, MPI_INT, id + exp_2(m - 1), nowID, MPI_COMM_WORLD);
        }
    }
    /*数据接收（从进程）*/
    if (nowID == id + exp_2(m - 1)) {    //当前进程是负责接收的
        MPI_Recv(&length, 1, MPI_INT, id, id, MPI_COMM_WORLD, &status);
        if (length > 0) {   //id+2^(m-1)进程从id进程接收后部分数据
            t = (int*)malloc(length * sizeof(int));
            if (t == 0) printf("Malloc memory error!");
            MPI_Recv(t, length, MPI_INT, id, id, MPI_COMM_WORLD, &status);
        }
    }
    /*递归排序*/
    j = r - 1 - start;
    MPI_Bcast(&j, 1, MPI_INT, id, MPI_COMM_WORLD);//MPI_Bcast 由 id 进程发起，所有进程（包括 id 和 id + 2^(m-1)）都会收到 j（前半部分的长度）。
    if (j > 0)     //负责分发的进程的数据不为空
        paraQuickSort(data, start, r - 1, m - 1, id, nowID, N);   //递归调用快排函数，对前部分数据进行排序
    j = length;
    MPI_Bcast(&j, 1, MPI_INT, id, MPI_COMM_WORLD);//MPI_Bcast 再次由 id 进程发起，所有进程收到 j（后半部分的长度）
    if (j > 0)     //负责接收的进程的数据不为空
        paraQuickSort(t, 0, length - 1, m - 1, id + exp_2(m - 1), nowID, N);   //递归调用快排函数，对后部分数据进行排序
    /*结果归并*/
    if ((nowID == id + exp_2(m - 1)) && (length > 0))     //id+2^(m-1)进程发送结果给id进程
        MPI_Send(t, length, MPI_INT, id, id + exp_2(m - 1), MPI_COMM_WORLD);
    if ((nowID == id) && id + exp_2(m - 1) < N && (length > 0))     //id进程接收id+2^(m-1)进程发送的结果
        MPI_Recv(data + r + 1, length, MPI_INT, id + exp_2(m - 1), id + exp_2(m - 1), MPI_COMM_WORLD, &status);
}

int main(int argc, char* argv[])
{
    int* data;
    int rank, size;
    int i, j, m, r, n = ELEMENTS;   //随机数组的长度
    double start_time, end_time;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  //当前进程的进程号
    MPI_Comm_size(MPI_COMM_WORLD, &size);  //总进程数

    if (rank == 0) {   //根进程生成随机数组
        data = (int*)malloc(n * sizeof(int));
        read_random_array(data, "random_array.bin");  //读取随机数组
    }
    m = log_2(size);  //第一次分发需要给第2^(m-1)个进程
    start_time = MPI_Wtime();
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);  //广播n
    paraQuickSort(data, 0, n - 1, m, 0, rank, size);  //执行快排
    end_time = MPI_Wtime();

    if (rank == 0) {   //根进程输出并行时间
        for (i = 0; i < n && i < 10; i++)   //n太大时只输出前10个
            printf("%d ", data[i]);
        printf("\n并行时间：%lfs\n", end_time - start_time);
    }
    MPI_Finalize();
}

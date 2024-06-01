#include<iostream>
#include<iomanip>
#include<semaphore.h>
#include <sys/time.h>
#include "mpi.h"
#include<unistd.h>
#include<cstring>
using namespace std;

const int maxSize = 1024;
float A[maxSize][maxSize];

void generateSample(int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < i; j++) {
			A[i][j] = 0;//下三角赋值为0;
		}
		A[i][i] = 1.0;//对角线赋值为1;
		for (int j = i; j < N; j++) {
			A[i][j] = rand();//上三角赋值为任意值;
		}
	}
	for (int k = 0; k < N; k++) {
		for (int i = k + 1; i < N; i++) {
			for (int j = 0; j < N; j++) {
				A[i][j] += A[k][j];//每一行都加上比自己下标小的行;
			}
		}
	}
}
void show(int N) {//打印结果;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << fixed << setprecision(0) << A[i][j] << " ";
		}
		cout << endl;
	}
}

//串行算法:
void serialSolution(int N) {
	for (int k = 0; k < N; k++) {
		for (int j = k + 1; j < N; j++) {
			A[k][j] /= A[k][k];
		}
		A[k][k] = 1.0;
		for (int i = k + 1; i < N; i++) {
			for (int j = k + 1; j < N; j++) {
				A[i][j] -= A[i][k] * A[k][j];
			}
			A[i][k] = 0;
		}
	}
}

//并行算法:
void parallelSolution(int N) {

    int myid, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    int r1, r2;
    int num = N / numprocs;
    r1 = myid * num;
    if(myid == numprocs - 1) {
		r2 = (myid + 1) * N / numprocs;
	} else {
		r2 = (myid + 1) * num;
	}

    for(int k = 0; k < N; k++ ) {
        if(k >= r1 && k < r2) {
            for(int j = k + 1; j < N; j++) {
				A[k][j] = A[k][j] / A[k][k];
			}
            A[k][k] = 1;
            for(int j = 0; j < numprocs; j++) {
                if(j != myid)
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, k, MPI_COMM_WORLD);
            }
        } else {
			MPI_Recv(&A[k][0], N, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,MPI_STATUSES_IGNORE);
		}
        for(int i=r1; i<r2; i++) {
            if(i <= k) continue; //做完除法的行不需要再进行消去操作。
            for(int j = k + 1; j < N; j++) {
				A[i][j] = A[i][j]-A[k][j]*A[i][k];
			}
            A[i][k] = 0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}


int main(int argc, char *argv[]) {

	MPI_Init(&argc, &argv);

	int N = 1024;
	int step = 64;
	struct timeval start1;
	struct timeval end1;
	struct timeval start2;
	struct timeval end2;
	cout.flags(ios::left);

    for (int i = step; i <= N; i += step) {

		int myid;
    	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

		//串行算法
		generateSample(i);
		if(myid == 0) gettimeofday(&start1, NULL);
		if(myid == 0) serialSolution(i);
		if(myid == 0) gettimeofday(&end1, NULL);

		//并行算法:
		generateSample(i);
		if(myid == 0) gettimeofday(&start2, NULL);
		parallelSolution(i);
		if(myid == 0) gettimeofday(&end2, NULL);

		//用时统计:
		float time1 = (end1.tv_sec - start1.tv_sec) + float((end1.tv_usec - start1.tv_usec)) / 1000000;//单位s;
		float time2 = (end2.tv_sec - start2.tv_sec) + float((end2.tv_usec - start2.tv_usec)) / 1000000;//单位s;

		if(myid == 0) {
			cout << fixed << setprecision(6);
        	cout << setw(13) << "数组规模" <<  i << ": " << "单线程平均用时：" << setw(20) << time1 << endl;
        	cout << setw(13) << " " << "MPI平均用时：" << setw(20) << time2 << endl;
        	cout << endl;
			//cout << time1/time2 << endl;
		}
	}
	MPI_Finalize();
	return 0;
}

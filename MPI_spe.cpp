#include <iostream>
#include <fstream>
#include <iomanip>
#include <semaphore.h>
#include <sys/time.h>
#include "mpi.h"
#include <unistd.h>
#include <cstring>
using namespace std;

//单线程算法:
bool serialSolution(int selection) {
    //selection 决定读取哪个文件
    string Folders[] = { "1_130_22_8", "2_254_106_53", "3_562_170_53", "4_1011_539_263", "5_2362_1226_453",
    "6_3799_2759_1953","7_8399_6375_4535", "11_85401_5724_756" };
    struct Size {
        int a;
        int b;
        int c;//分别为矩阵列数，消元子个数和被消元行个数
    } fileSize[] = { {130, 22, 8}, {254, 106, 53}, {562, 170, 53}, {1011, 539, 262}, {2362, 1226, 453},
    {3799, 2759, 1953},{8399, 6375, 4535},{85401,5724,756} };

    ifstream erFile;
    ifstream eeFile;
    erFile.open("/home/data/Groebner/" + Folders[selection] + "/1.txt", std::ios::binary);//消元子文件
    eeFile.open("/home/data/Groebner/" + Folders[selection] + "/2.txt", std::ios::binary);//被消元行文件
    ofstream resFile("/home/data/Groebner/" + Folders[selection] + "/res_of_serial.txt", ios::trunc);//结果回写文件

    int COL = fileSize[selection].a;
    int erROW = fileSize[selection].b;
    int eeROW = fileSize[selection].c;
    int N = (COL + 31) / 32;

    int** ER = new int* [erROW];
    int** EE = new int* [eeROW];
    int* flag = new int[COL] {0};

    //读取消元子:
    for (int i = 0; i < erROW; i++) {
        ER[i] = new int[N] {0};
        int col;
        char ch = ' ';
        erFile >> col;
        int r = COL - 1 - col;
        ER[i][r >> 5] = 1 << (31 - (r & 31));
        erFile.get(ch);
        flag[col] = i + 1;
        while (erFile.peek() != '\r') {
            erFile >> col;
            int diff = COL - 1 - col;
            ER[i][diff >> 5] += 1 << (31 - (diff & 31));
            erFile.get(ch);
        }
    }

    //读取被消元行:
    for (int i = 0; i < eeROW; i++) {
        EE[i] = new int[N] {0};
        int col;
        char ch = ' ';
        while (eeFile.peek() != '\r') {
            eeFile >> col;
            int diff = COL - 1 - col;
            EE[i][diff >> 5] += 1 << (31 - (diff & 31));
            eeFile.get(ch);
        }
        eeFile.get(ch);
    }

    for (int i = 0; i < eeROW; i++) {
        int byte = 0;
        int bit = 0;
        int N = (COL + 31) / 32;
        while (true) {
            while (byte < N && EE[i][byte] == 0) {
                byte++;
                bit = 0;
            }
            if (byte >= N) {
                break;
            }
            int temp = EE[i][byte] << bit;
            while (temp >= 0) {
                bit++;
                temp <<= 1;
            }
            int& isExist = flag[COL - 1 - (byte << 5) - bit];
            if (!isExist == 0) {
                int* er = isExist > 0 ? ER[isExist - 1] : EE[~isExist];
                for (int j = 0; j < N; j++) {
                    EE[i][j] ^= er[j];
                }
            }
            else {
                isExist = ~i;
                break;
            }
        }
    }

    //将得到的结果写回到文件中
    for (int i = 0; i < eeROW; i++) {
        int count = COL - 1;
        for (int j = 0; j < N; j++) {
            int dense = EE[i][j];
            for (int k = 0; k < 32; k++) {
                if (dense == 0) {
                    break;
                }
                else if (dense < 0) {
                    resFile << count - k << ' ';
                }
                dense <<= 1;
            }
            count -= 32;
        }
        resFile << '\n';
    }
    //释放空间:
    for (int i = 0; i < erROW; i++) {
        delete[] ER[i];
    }
    delete[] ER;

    for (int i = 0; i < eeROW; i++) {
        delete[] EE[i];
    }
    delete[] EE;
    delete[] flag;
    return true;
}

//多进程算法:
bool parallelSolution(int selection) {
    //selection 决定读取哪个文件
    string Folders[] = { "1_130_22_8", "2_254_106_53", "3_562_170_53", "4_1011_539_263", "5_2362_1226_453",
    "6_3799_2759_1953","7_8399_6375_4535", "11_85401_5724_756" };
    struct Size {
        int a;
        int b;
        int c;//分别为矩阵列数，消元子个数和被消元行个数
    } fileSize[] = { {130, 22, 8}, {254, 106, 53}, {562, 170, 53}, {1011, 539, 262}, {2362, 1226, 453},
    {3799, 2759, 1953},{8399, 6375, 4535},{85401,5724,756} };

    ifstream erFile;
    ifstream eeFile;
    erFile.open("/home/data/Groebner/" + Folders[selection] + "/1.txt", std::ios::binary);//消元子文件
    eeFile.open("/home/data/Groebner/" + Folders[selection] + "/2.txt", std::ios::binary);//被消元行文件
    ofstream resFile("/home/data/Groebner/" + Folders[selection] + "/res_of_parallel.txt", ios::trunc);//结果回写文件

    int COL = fileSize[selection].a;
    int erROW = fileSize[selection].b;
    int eeROW = fileSize[selection].c;
    int N = (COL + 31) / 32;

    int** ER = new int* [erROW];
    int** EE = new int* [eeROW];
    int* flag = new int[COL] {0};

    //读取消元子:
    for (int i = 0; i < erROW; i++) {
        ER[i] = new int[N] {0};
        int col;
        char ch = ' ';
        erFile >> col;
        int r = COL - 1 - col;
        ER[i][r >> 5] = 1 << (31 - (r & 31));
        erFile.get(ch);
        flag[col] = i + 1;
        while (erFile.peek() != '\r') {
            erFile >> col;
            int diff = COL - 1 - col;
            ER[i][diff >> 5] += 1 << (31 - (diff & 31));
            erFile.get(ch);
        }
    }

    //读取被消元行:
    for (int i = 0; i < eeROW; i++) {
        EE[i] = new int[N] {0};
        int col;
        char ch = ' ';
        while (eeFile.peek() != '\r') {
            eeFile >> col;
            int diff = COL - 1 - col;
            EE[i][diff >> 5] += 1 << (31 - (diff & 31));
            eeFile.get(ch);
        }
        eeFile.get(ch);
    }


    int myid, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Request request;

    int r1, r2;
    int num = N / numprocs;
    r1 = myid * num;
    if(myid == numprocs - 1) {
		r2 = (myid + 1) * N / numprocs;
	} else {
		r2 = (myid + 1) * num;
	}

    for (int i = r1; i < r2; i += 1) {
        int byte = 0;
        int bit = 0;
        int N = (COL + 31) / 32;
        while (true) {
            while (byte < N && EE[i][byte] == 0) {
                byte++;
                bit = 0;
            }
            if (byte >= N) {
                break;
            }
            int temp = EE[i][byte] << bit;
            while (temp >= 0) {
                bit++;
                temp <<= 1;
            }
            int& isExist = flag[COL - 1 - (byte << 5) - bit];
            if (!isExist == 0) {
                int* er = isExist > 0 ? ER[isExist - 1] : EE[~isExist];
                for (int j = 0; j < N; j++) {
                    EE[i][j] ^= er[j];
                }
            }
            else {
                isExist = ~i;
                MPI_Ibcast(&EE[i][0], N, MPI_FLOAT, myid, MPI_COMM_WORLD, &request);
                MPI_Ibcast(&flag[COL - 1 - (byte << 5) - bit], 1, MPI_FLOAT, myid, MPI_COMM_WORLD, &request);
                break;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //将得到的结果写回到文件中
    for (int i = 0; i < eeROW; i++) {
        int count = COL - 1;
        for (int j = 0; j < N; j++) {
            int dense = EE[i][j];
            for (int k = 0; k < 32; k++) {
                if (dense == 0) {
                    break;
                }
                else if (dense < 0) {
                    resFile << count - k << ' ';
                }
                dense <<= 1;
            }
            count -= 32;
        }
        resFile << '\n';
    }
    //释放空间:
    for (int i = 0; i < erROW; i++) {
        delete[] ER[i];
    }
    delete[] ER;

    for (int i = 0; i < eeROW; i++) {
        delete[] EE[i];
    }
    delete[] EE;
    delete[] flag;
    return true;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    struct timeval start1;
    struct timeval end1;
    struct timeval start2;
    struct timeval end2;
    cout.flags(ios::left);
    for (int i = 0; i <= 7; i += 1) { //遍历文件:
        //传统算法
        if(myid == 0) gettimeofday(&start1, NULL);
        serialSolution(i);
        if(myid == 0) gettimeofday(&end1, NULL);


        //多线程算法:
        if(myid == 0) gettimeofday(&start2, NULL);
        parallelSolution(i);
        if(myid == 0) gettimeofday(&end2, NULL);

        //用时统计:
        float time1 = (end1.tv_sec - start1.tv_sec) + float((end1.tv_usec - start1.tv_usec)) / 1000000;//单位s;
        float time2 = (end2.tv_sec - start2.tv_sec) + float((end2.tv_usec - start2.tv_usec)) / 1000000;//单位s;

        if(myid == 0) {
            // cout << fixed << setprecision(6);
            // cout << setw(10) << "数据集" <<  i << ": " << "单线程平均用时：" << setw(20) << time1 << endl;
            // cout << setw(10) << " " << "MPI平均用时：" << setw(20) << time2 << endl;
            // cout << endl;
            cout << time1 <<" "<< time2 << endl;
        }
    }
    MPI_Finalize();
    return 0;
}

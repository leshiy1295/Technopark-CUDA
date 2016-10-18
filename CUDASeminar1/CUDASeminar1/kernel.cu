/*
	ДЗ1: 
		- gnuplot;
		- версия для CPU + сравнение;
		- задание №2
		*- впоследствии будет ещё неявная схема, и надо оптимизировать
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include <iostream>

#define CUDA_CALL(x) do { \
		cudaError_t cudaStatus = (x); \
		if (cudaStatus != cudaSuccess) { \
			printf("Cuda Failed with error %s\n", cudaGetErrorString(cudaStatus)); \
			system("pause"); \
			exit(cudaStatus); \
		} \
	} while (0);

#define PRINT
//#define COMPARE

#define BLOCK_SIZE 32

#define T 5
#define L 10

#define xPoints (200 + 1)
#define tPoints (10000 + 1)

#define DT T * 1.0 / (tPoints - 1)
#define DX L * 1.0 / (xPoints - 1)

__global__ void computeTemp(double *temp, const int k)
{
	int threadId = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadId == xPoints - 1) {
		temp[k*xPoints + threadId] = temp[(k - 1) * xPoints + threadId] + DT;
	}
	else if (threadId < xPoints - 1 && threadId > 0) {
		temp[k*xPoints + threadId] = (
				temp[(k - 1) * xPoints + threadId + 1] -
				2 * temp[(k - 1) * xPoints + threadId] +
				temp[(k - 1) * xPoints + threadId - 1]
			) * DT / (DX * DX) + temp[(k - 1) * xPoints + threadId];
	}
}

int main()
{
	int totalElemCount = xPoints * tPoints;
	int memSize = totalElemCount * sizeof(double);

	double *temp = (double *)calloc(totalElemCount, sizeof(double));
	double *devTemp;
	
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaDeviceReset());
	CUDA_CALL(cudaMalloc(&devTemp, memSize));
	CUDA_CALL(cudaMemcpy(devTemp, temp, memSize, cudaMemcpyHostToDevice));

	int blocksCount = xPoints / BLOCK_SIZE;
	if (xPoints % BLOCK_SIZE != 0) {
		++blocksCount;
	}

	for (int k = 1; k < tPoints; ++k) {
		computeTemp<<<blocksCount, BLOCK_SIZE>>>(devTemp, k);
	}

	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaMemcpy(temp, devTemp, memSize, cudaMemcpyDeviceToHost));
	
#ifdef PRINT
	for (int i = 0; i < tPoints; ++i) {
		for (int j = 0; j < xPoints; ++j) {
			printf("%f %f\n", j * DX, temp[i * xPoints + j]);
		}
		if (i != tPoints - 1) {
			printf("\n\n");
		}
}
#endif // PRINT
	
#ifdef COMPARE

#endif // COMPARE


	CUDA_CALL(cudaFree(devTemp));
	CUDA_CALL(cudaDeviceReset());
	free(temp);

    return 0;
}

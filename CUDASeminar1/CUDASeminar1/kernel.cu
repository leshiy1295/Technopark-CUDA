/*
	ДЗ: 
		- gnuplot;
		- версия для CPU + сравнение;
		- задание №2;
		- неявная схема;
		- оптимизации по возможности.
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include <iostream>
#include <time.h>

#define CUDA_CALL(x) do { \
		cudaError_t cudaStatus = (x); \
		if (cudaStatus != cudaSuccess) { \
			printf("Cuda Failed with error %s\n", cudaGetErrorString(cudaStatus)); \
			system("pause"); \
			exit(cudaStatus); \
		} \
	} while (0);

//#define PRINT
//#define COMPARE
//#define EXPLICIT
#define IMPLICIT

#ifdef IMPLICIT
#define EPS 1.e-3
#endif

#define BLOCK_SIZE 32

#define L 10

#define xPoints (10 + 1)
#define tPoints (5 + 1)

#define DT 5
#define DX L * 1.0 / (xPoints - 1)

__global__ void computeTemp(double *temp, const int k, double *delta)
{
	int threadId = threadIdx.x + blockDim.x * blockIdx.x;
#ifdef EXPLICIT
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
#endif
#ifdef IMPLICIT
	// одна итерация
	if (threadId == xPoints - 1) {
		temp[k*xPoints + threadId] = temp[(k - 1) * xPoints + threadId] + DT;
	}
	else {
		if (threadId < xPoints - 1 && threadId > 0) {
			delta[threadId] = temp[k * xPoints + threadId];
			temp[k * xPoints + threadId] = (
				temp[(k - 1) * xPoints + threadId] +
				DT / (DX * DX) * temp[(k - 1) * xPoints + threadId - 1] +
				DT / (DX * DX) * temp[(k - 1) * xPoints + threadId + 1]
				) / (2 * DT / (DX * DX) + 1);
			delta[threadId] = temp[k * xPoints + threadId] - delta[threadId];
		}
	}
#endif
}

int main()
{
#ifdef COMPARE
	cudaEvent_t GPUStartWithMem, GPUStartKernelOnly, GPUStopWithMem, GPUStopKernelOnly;
	float CPUStart, CPUStop;
	
	float GPUTimeWithMem = 0.0f;
	float GPUTimeKernelOnly = 0.0f;
	float CPUTime = 0.0f;
#endif

	int totalElemCount = xPoints * tPoints;
	int memSize = totalElemCount * sizeof(double);

	double *temp = (double *)calloc(totalElemCount, sizeof(double));
	double *devTemp;

	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaDeviceReset());
	CUDA_CALL(cudaMalloc(&devTemp, memSize));

#ifdef IMPLICIT
	double *devBuffer;
	CUDA_CALL(cudaMalloc(&devBuffer, xPoints * sizeof(double)));
#endif

#ifdef COMPARE
	CUDA_CALL(cudaEventCreate(&GPUStartKernelOnly));
	CUDA_CALL(cudaEventCreate(&GPUStopKernelOnly));
	CUDA_CALL(cudaEventCreate(&GPUStartWithMem));
	CUDA_CALL(cudaEventCreate(&GPUStopWithMem));

	CUDA_CALL(cudaEventRecord(GPUStartWithMem, 0));
#endif

	CUDA_CALL(cudaMemcpy(devTemp, temp, memSize, cudaMemcpyHostToDevice));

	int blocksCount = xPoints / BLOCK_SIZE;
	if (xPoints % BLOCK_SIZE != 0) {
		++blocksCount;
	}

#ifdef COMPARE
	CUDA_CALL(cudaEventRecord(GPUStartKernelOnly, 0));
#endif

	for (int k = 1; k < tPoints; ++k) {
#ifdef IMPLICIT
		CUDA_CALL(cudaMemset(&devBuffer, 0, xPoints * sizeof(double)));
		computeTemp << <blocksCount, BLOCK_SIZE >> >(devTemp, k, devBuffer);
#endif
#ifdef EXPLICIT
		computeTemp << <blocksCount, BLOCK_SIZE >> >(devTemp, k);
#endif
		CUDA_CALL(cudaDeviceSynchronize());
	}

	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());

#ifdef COMPARE
	CUDA_CALL(cudaEventRecord(GPUStopKernelOnly, 0));
#endif

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
	CUDA_CALL(cudaEventRecord(GPUStopWithMem, 0));
	CUDA_CALL(cudaEventSynchronize(GPUStopWithMem));
	CUDA_CALL(cudaEventElapsedTime(&GPUTimeWithMem, GPUStartWithMem, GPUStopWithMem));
	CUDA_CALL(cudaEventElapsedTime(&GPUTimeKernelOnly, GPUStartKernelOnly, GPUStopKernelOnly));
	printf("GPU Time With Mem: %.3f ms\n", GPUTimeWithMem);
	printf("GPU Time Kernel Only: %.3f ms\n", GPUTimeKernelOnly);

	CUDA_CALL(cudaEventDestroy(GPUStartKernelOnly));
	CUDA_CALL(cudaEventDestroy(GPUStopKernelOnly));
	CUDA_CALL(cudaEventDestroy(GPUStartWithMem));
	CUDA_CALL(cudaEventDestroy(GPUStopWithMem));

	CPUStart = clock();

#ifdef EXPLICIT
	for (int k = 1; k < tPoints; ++k) {
		for (int j = 0; j < xPoints - 1; ++j) {
				temp[k*xPoints + j] = (
					temp[(k - 1) * xPoints + j + 1] -
					2 * temp[(k - 1) * xPoints + j] +
					temp[(k - 1) * xPoints + j - 1]
					) * DT / (DX * DX) + temp[(k - 1) * xPoints + j];
		}
		temp[k*xPoints + xPoints - 1] = temp[(k - 1) * xPoints + xPoints - 1] + DT;
	}
#endif
#ifdef IMPLICIT
	for (int k = 1; k < tPoints; ++k) {

	}
#endif

	CPUStop = clock();
	CPUTime = 1000. * (CPUStop - CPUStart) / CLOCKS_PER_SEC;
	printf("CPU time: %.3f ms\n", CPUTime);

#endif // COMPARE

	CUDA_CALL(cudaFree(devTemp));
	CUDA_CALL(cudaDeviceReset());
	free(temp);

    return 0;
}

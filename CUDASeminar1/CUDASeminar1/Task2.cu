/*
	Моделирование воздействия на прямоугольную мембрану
	COMPARE - режим сравнения реализации на GPU и на CPU
	PRINT - режим вывода значений координат мембраны
	EXPLICIT - явный метод получения результата
	IMPLICIT - неявный метод получения результата
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

#define PRINT
//#define COMPARE
#define EXPLICIT
//#define IMPLICIT

#ifdef IMPLICIT
#define EPS 1.e-3
#endif

#define BLOCK_SIZE 32

#define a 5
#define F(x, y, t) 10

#define X 500
#define Y 500

#define xPoints (50 + 1)
#define yPoints (50 + 1)
#define tPoints (1000 + 1)

#define DT 1
#define DX X * 1.0 / (xPoints - 1)
#define DY Y * 1.0 / (yPoints - 1)

#ifdef EXPLICIT
__global__ void computeZ(double *z, const int k)
#endif
#ifdef IMPLICIT
__global__ void computeZ(double *z, const int k, double *delta)
#endif
{
	int threadId = threadIdx.x + blockDim.x * blockIdx.x;
	// threadId - тут рассматривается ещё и как линейный адрес матрицы размерности xPoints * yPoints
	if (threadId >= xPoints * yPoints) {
		return;
	}

#ifdef EXPLICIT
	int x = threadId / yPoints;
	int y = threadId % yPoints;
	// таким образом, представим threadId = x * yPoints + y
	// краевые условия - на границе значение z равно 0
	if (x == 0 || y == 0 || x == xPoints - 1 || y == yPoints - 1) {
		z[k * xPoints * yPoints + threadId] = 0;
	}
	else {
		z[k * xPoints * yPoints + x * yPoints + y] = DT * DT * (
				F(x, y, k) + a * a * (
					(
						z[(k - 1) * xPoints * yPoints + (x + 1) * yPoints + y]
						- 2 * z[(k - 1) * xPoints * yPoints + x * yPoints + y]
						+ z[(k - 1) * xPoints * yPoints + (x - 1) * yPoints + y]
					) / (DX * DX)
					+
					(
						z[(k - 1) * xPoints * yPoints + x * yPoints + y + 1]
						- 2 * z[(k - 1) * xPoints * yPoints + x * yPoints + y]
						+ z[(k - 1) * xPoints * yPoints + x * yPoints + y - 1]
					) / (DY * DY)
				)
			)
			- z[(k - 2) * xPoints * yPoints + x * yPoints + y]
			+ 2 * z[(k - 1) * xPoints * yPoints + x * yPoints + y];
	}
#endif
#ifdef IMPLICIT
	// одна итерация
	if (threadId == xPoints - 1) {
		z[k*xPoints + threadId] = z[(k - 1) * xPoints + threadId] + DT;
	}
	else {
		if (threadId < xPoints - 1 && threadId > 0) {
			delta[threadId] = z[k * xPoints + threadId];
			z[k * xPoints + threadId] = (
				z[(k - 1) * xPoints + threadId] +
				DT / (DX * DX) * temp[(k - 1) * xPoints + threadId - 1] +
				DT / (DX * DX) * temp[(k - 1) * xPoints + threadId + 1]
				) / (2 * DT / (DX * DX) + 1);
			delta[threadId] = abs(z[k * xPoints + threadId] - delta[threadId]);
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

	int totalElemCount = xPoints * yPoints * (tPoints + 1); // нужны k - 2. Введём искусственно матрицу для -1
	int memSize = totalElemCount * sizeof(double);

	double *z = (double *)calloc(totalElemCount, sizeof(double));
	double *devZ;

	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaDeviceReset());
	CUDA_CALL(cudaMalloc(&devZ, memSize));

#ifdef IMPLICIT
	double *delta = (double *)calloc(xPoints, sizeof(double));
	double *devDelta;
	CUDA_CALL(cudaMalloc(&devDelta, xPoints * sizeof(double)));
#endif

#ifdef COMPARE
	CUDA_CALL(cudaEventCreate(&GPUStartKernelOnly));
	CUDA_CALL(cudaEventCreate(&GPUStopKernelOnly));
	CUDA_CALL(cudaEventCreate(&GPUStartWithMem));
	CUDA_CALL(cudaEventCreate(&GPUStopWithMem));

	CUDA_CALL(cudaEventRecord(GPUStartWithMem, 0));
#endif

	CUDA_CALL(cudaMemcpy(devZ, z, memSize, cudaMemcpyHostToDevice));

	int blocksCount = xPoints * yPoints / BLOCK_SIZE;
	if (xPoints * yPoints % BLOCK_SIZE != 0) {
		++blocksCount;
	}

#ifdef COMPARE
	CUDA_CALL(cudaEventRecord(GPUStartKernelOnly, 0));
#endif

	for (int k = 2; k < tPoints + 1; ++k) {
#ifdef IMPLICIT
		bool flag = false; // флаг сходимости решения СЛАУ
		while (!flag) {
			computeTemp << <blocksCount, BLOCK_SIZE >> > (devTemp, k, devDelta);
			CUDA_CALL(cudaMemcpy(delta, devDelta, xPoints * sizeof(double), cudaMemcpyDeviceToHost));
			double sum = 0;
			for (int i = 0; i < xPoints; ++i) {
				sum += delta[i];
			}
			if (sum < EPS) {
				flag = true;
			}
		}
#endif
#ifdef EXPLICIT
		computeZ << <blocksCount, BLOCK_SIZE >> >(devZ, k);
#endif
	}

	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());

#ifdef COMPARE
	CUDA_CALL(cudaEventRecord(GPUStopKernelOnly, 0));
#endif

	CUDA_CALL(cudaMemcpy(z, devZ, memSize, cudaMemcpyDeviceToHost));

#ifdef PRINT
	for (int k = 1; k < tPoints + 1; ++k) {
		for (int i = 0; i < xPoints; ++i) {
			for (int j = 0; j < yPoints; ++j) {
				printf("%f %f %f\n", i * DX, j * DY, z[k * xPoints * yPoints + i * yPoints + j]);
			}
		}
		if (k != tPoints) {
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
	memset(temp, 0, memSize);
#ifdef EXPLICIT
	for (int k = 1; k < tPoints; ++k) {
		temp[k*xPoints + xPoints - 1] = temp[(k - 1) * xPoints + xPoints - 1] + DT;
		for (int j = 0; j < xPoints - 1; ++j) {
			temp[k*xPoints + j] = (
				temp[(k - 1) * xPoints + j + 1] -
				2 * temp[(k - 1) * xPoints + j] +
				temp[(k - 1) * xPoints + j - 1]
				) * DT / (DX * DX) + temp[(k - 1) * xPoints + j];
		}
	}
#endif
#ifdef IMPLICIT
	for (int k = 1; k < tPoints; ++k) {
		bool flag = false; // флаг сходимости решения СЛАУ
		while (!flag) {
			temp[k*xPoints + xPoints - 1] = temp[(k - 1) * xPoints + xPoints - 1] + DT;
			for (int j = 1; j < xPoints - 1; ++j) {
				delta[j] = temp[k * xPoints + j];
				temp[k * xPoints + j] = (
					temp[(k - 1) * xPoints + j] +
					DT / (DX * DX) * temp[(k - 1) * xPoints + j - 1] +
					DT / (DX * DX) * temp[(k - 1) * xPoints + j + 1]
					) / (2 * DT / (DX * DX) + 1);
				delta[j] = abs(temp[k * xPoints + j] - delta[j]);
			}
			double sum = 0;
			for (int i = 0; i < xPoints; ++i) {
				sum += delta[i];
			}
			if (sum < EPS) {
				flag = true;
			}
		}
	}
#endif

	CPUStop = clock();
	CPUTime = 1000. * (CPUStop - CPUStart) / CLOCKS_PER_SEC;
	printf("CPU time: %.3f ms\n", CPUTime);

#endif // COMPARE

	CUDA_CALL(cudaFree(devZ));
	CUDA_CALL(cudaDeviceReset());
	free(z);

	return 0;
}

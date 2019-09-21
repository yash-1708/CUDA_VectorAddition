
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <ctime>
#define N 100000000//size of vectors declared globally
#define M 1024//threads per block

//main kernel that runs on GPU
__global__ void VecAdd(int* a, int* b, int* c) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	c[i] = a[i] + b[i];
	//printf("\n block: %d \t a:%d \t + \t b:%d \t = \t c:%d", blockIdx.x, a[i], b[i], c[i]);
}

void random_ints(int* arr, int size)
{
	int i;
	for (i = 0; i < size; ++i)
		arr[i] = rand() % 100;
}

int main() {
	clock_t start, stop;

	int *a_CPU, *b_CPU, *c_CPU, *d_CPU; //CPU vectors pointers
	//memory allocation in CPU
	a_CPU = (int*)malloc(N * sizeof(int));
	b_CPU = (int*)malloc(N * sizeof(int));
	c_CPU = (int*)malloc(N * sizeof(int));
	d_CPU = (int*)malloc(N * sizeof(int));

	int *a_GPU, *b_GPU, *c_GPU; //pointers to be stored in GPU
	//memory allocation in GPU
	cudaMalloc((void**)&a_GPU, N * sizeof(int));
	cudaMalloc((void**)&b_GPU, N * sizeof(int));
	cudaMalloc((void**)&c_GPU, N * sizeof(int));

	//vector data allocation
	random_ints(a_CPU, N);
	random_ints(b_CPU, N);
	
	//copy CPU arry references into GPU pointers
	cudaMemcpy(a_GPU, a_CPU, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b_GPU, b_CPU, N * sizeof(int), cudaMemcpyHostToDevice);

	//kernel call
	start = std::clock();
	VecAdd<<<(N+M-1)/M,M>>>(a_GPU, b_GPU, c_GPU);
	cudaDeviceSynchronize();
	stop = std::clock();
	long float timeP = stop - start;

	//copy result from GPU to CPU
	cudaMemcpy(c_CPU, c_GPU, N * sizeof(int), cudaMemcpyDeviceToHost);

	//unparallel operation
	start = std::clock();
	for (size_t i = 0; i < N; i++)
	{
		d_CPU[i] = a_CPU[i] + b_CPU[i];
	}
	stop = std::clock();
	long float timeN = stop - start;
	
	//getting GPU properties and storing in prop
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int cores = prop.multiProcessorCount * 128;
	float totalCost = cores * timeP;

	//results printing
	printf("\n********************************************************************************************************\n");
	printf("N \t\t\t Nor Time \t Par Time \t Cores \t\t Tot Cost \t Speedup \t Efficiency \n");
	printf("%-20d \t %-7.3f \t %-7.3f \t %-10d \t %-7.3f \t %-7.3f \t %-5.5f \n", N, timeN, timeP, cores, totalCost, timeN / timeP, timeN / (timeP * cores));
	printf("\n********************************************************************************************************\n");

	//free memory
	cudaFree(a_GPU);
	cudaFree(b_GPU);
	cudaFree(c_GPU);
	free(a_CPU);
	free(b_CPU);
	free(c_CPU);
	free(d_CPU);

	return 0;
}
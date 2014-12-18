#include "stdio.h"
#include "stdlib.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

#define BLOCK_HEIGHT 64 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void switchRows(float* A, float* d_rowk, float* d_rowi, int* rightColumnIndices, int dim, int i, int k){
	int j = blockIdx.x*blockDim.x+threadIdx.x;
	int j_unshifted = blockIdx.x*blockDim.x+threadIdx.x;
	int temp_indexi;
	int temp_indexk;
	if (k!=i){
		temp_indexi=rightColumnIndices[i];
		temp_indexk=rightColumnIndices[k];
		rightColumnIndices[i]=temp_indexk;
		rightColumnIndices[k]=temp_indexi;
	}
	if (j+i>dim-1){
		j=dim-i+rightColumnIndices[j-dim+i];
	}
	A[(j+i)*dim+k]=d_rowk[j_unshifted]; // I store the line I am not interested in now
	A[(j+i)*dim+i]=d_rowi[j_unshifted];
	if (k!=i){
		A[(dim+temp_indexi)*dim+i]=0;
		A[(dim+temp_indexi)*dim+k]=1;
		A[(dim+temp_indexk)*dim+i]=1;
		A[(dim+temp_indexk)*dim+k]=0;
	}
}

__global__ void createIndexVector(int* rightColumnIndices, int dim){
	int j = blockIdx.x*blockDim.x+threadIdx.x;
	rightColumnIndices[j]=j;
}

__global__ void storeRows(float* A, float* d_rowk, float* d_rowi, int* rightColumnIndices, int dim, int i, int k){
	int j = blockIdx.x*blockDim.x+threadIdx.x;
	int j_unshifted = blockIdx.x*blockDim.x+threadIdx.x;
	if (j+i>dim-1){
		j=dim-i+rightColumnIndices[j-dim+i];
	}
	d_rowk[j_unshifted]=A[(j+i)*dim+i];
	d_rowi[j_unshifted]=A[(j+i)*dim+k];
}

__global__ void workRow(float* A, int* rightColumnIndices, int dim, int i){
	__shared__ float Aii;
	__shared__ float rowi[BLOCK_HEIGHT]; //tohle muzes vyhnat az na 1024!!!
	Aii=A[i*dim+i];
	int j = blockIdx.x*blockDim.x+threadIdx.x;
	int j0 =threadIdx.x;
	if (j+i>dim-2){
		j=dim-i-1+rightColumnIndices[j-dim+i+1];
	}
	rowi[j0]=A[(j+i+1)*dim+i];
	__syncthreads();
	rowi[j0]=rowi[j0]/Aii;
	A[(j+i+1)*dim+i]=rowi[j0];
/*	if (abs(Aii)<0.001) printf("%f\n", Aii);*/
}


__global__ void workRows(float* A, int* rightColumnIndices, int dim, int ipiv){
	__shared__ float colpiv[BLOCK_HEIGHT];
	__shared__ float colj[BLOCK_HEIGHT];
	__shared__ float colj_piv;
	int i = blockIdx.y*blockDim.x+threadIdx.x;
	int i0 = threadIdx.x;
	int j = blockIdx.x;
	if (j+ipiv>dim-2){
		j=dim-ipiv-1+rightColumnIndices[j-dim+ipiv+1];
	}
	colpiv[i0]=A[ipiv*dim+i];
	colj[i0]=A[(j+ipiv+1)*dim+i];
	colj_piv=A[(j+ipiv+1)*dim+ipiv];
	if (i != ipiv){
		colj[i0]=colj[i0]-colj_piv*colpiv[i0];
	}
	A[(j+ipiv+1)*dim+i]=colj[i0];
}

__global__ void createPivotVector(float* A, float* v, int dim, int ipiv){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i<ipiv){
		v[i]=0;
	} else {
		v[i]=A[ipiv*dim+i];
	}
}

extern "C" void kernel_wrapper_(float* A, int* dim){
	size_t size=2**dim**dim*sizeof(float);
	size_t sizecol=*dim*sizeof(float);
	size_t sizerow=*dim*sizeof(float);
	size_t sizerowint=*dim*sizeof(int);
	float* d_A;
	float* d_col;
	float* d_rowi;
	float* d_rowk;
	int* d_indices;
	int max_idx;
	int strong_pivoting=1;
	cublasHandle_t handle;
	cublasStatus_t stat;

	cudaMalloc(&d_A,size);
	cudaMalloc(&d_col,sizecol);
	cudaMalloc(&d_rowi,sizerow);
	cudaMalloc(&d_rowk,sizerow);
	cudaMalloc(&d_indices,sizerowint);
	gpuErrchk( cudaMemcpy(d_A,A, size,cudaMemcpyHostToDevice) );
	cublasCreate(&handle);
	
	dim3 dimBlockRow(BLOCK_HEIGHT,1,1);
	dim3 dimGridRow(*dim/BLOCK_HEIGHT,1,1);
	dim3 dimBlockCol(BLOCK_HEIGHT,1,1);
	dim3 dimGridCol(*dim,*dim/BLOCK_HEIGHT,1);
	dim3 dimBlockPiv(BLOCK_HEIGHT,1,1);
	dim3 dimGridPiv(*dim/BLOCK_HEIGHT,1,1);
	dim3 dimBlockStoreRow(BLOCK_HEIGHT,1,1);
	dim3 dimGridStoreRow(*dim/BLOCK_HEIGHT,1,1);
	createIndexVector<<<dimGridRow,dimBlockRow>>>(d_indices,*dim);
	for (int i=0; i<*dim;++i){
		if (strong_pivoting == 1){
			createPivotVector<<<dimGridPiv,dimBlockPiv>>>(d_A,d_col,*dim,i);
			stat = cublasIsamax(handle, *dim, d_col, 1, &max_idx);
			if (stat != CUBLAS_STATUS_SUCCESS) printf("Max failed\n");
			storeRows<<<dimGridStoreRow,dimBlockStoreRow>>>(d_A,d_rowk,d_rowi,d_indices,*dim,i,max_idx-1);
			switchRows<<<dimGridStoreRow,dimBlockStoreRow>>>(d_A,d_rowk,d_rowi,d_indices,*dim,i,max_idx-1);
		} else{
        	max_idx=i+1;
		}
		workRow<<<dimGridRow,dimBlockRow>>>(d_A,d_indices,*dim,i);
		workRows<<<dimGridCol,dimBlockCol>>>(d_A,d_indices,*dim,i);
	}
    cublasDestroy(handle);
	gpuErrchk( cudaMemcpy(A,d_A, size,cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaFree(d_A) );
	gpuErrchk( cudaFree(d_col) );
	gpuErrchk( cudaFree(d_rowi) );
	gpuErrchk( cudaFree(d_rowk) );
	gpuErrchk( cudaFree(d_indices) );

	return;

}
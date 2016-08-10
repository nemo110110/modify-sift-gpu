////////////////////////////////////////////////////////////////////////////
//	File:		ProgramCU.cu
//	Author:		Changchang Wu
//	Description : implementation of ProgramCU and all CUDA kernels
//
//	Copyright (c) 2007 University of North Carolina at Chapel Hill
//	All Rights Reserved
//
//	Permission to use, copy, modify and distribute this software and its
//	documentation for educational, research and non-profit purposes, without
//	fee, and without a written agreement is hereby granted, provided that the
//	above copyright notice and the following paragraph appear in all copies.
//	
//	The University of North Carolina at Chapel Hill make no representations
//	about the suitability of this software for any purpose. It is provided
//	'as is' without express or implied warranty. 
//
//	Please send BUG REPORTS to ccwu@cs.unc.edu
//
////////////////////////////////////////////////////////////////////////////

#if defined(CUDA_SIFTGPU_ENABLED)

#include "GL/glew.h"
#include "stdio.h"

#include "CuTexImage.h"
#include "ProgramCU.h"
#include "GlobalUtil.h"

//----------------------------------------------------------------
//Begin SiftGPU setting section.
//////////////////////////////////////////////////////////
#define IMUL(X,Y) __mul24(X,Y)
//#define FDIV(X,Y) ((X)/(Y))
#define FDIV(X,Y) __fdividef(X,Y)

/////////////////////////////////////////////////////////
//filter kernel width range (don't change this)
#define KERNEL_MAX_WIDTH 33
#define KERNEL_MIN_WIDTH 5

//////////////////////////////////////////////////////////
//horizontal filter block size (32, 64, 128, 256, 512)
#define FILTERH_TILE_WIDTH 128
//#define FILTERH_TILE_WIDTH 256
//#define FILTERH_TILE_WIDTH 160
//thread block for vertical filter. FILTERV_BLOCK_WIDTH can be (4, 8 or 16)
#define FILTERV_BLOCK_WIDTH 16
#define FILTERV_BLOCK_HEIGHT 32
//The corresponding image patch for a thread block
#define FILTERV_PIXEL_PER_THREAD 4
#define FILTERV_TILE_WIDTH FILTERV_BLOCK_WIDTH
#define FILTERV_TILE_HEIGHT (FILTERV_PIXEL_PER_THREAD * FILTERV_BLOCK_HEIGHT)


//////////////////////////////////////////////////////////
//thread block size for computing Difference of Gaussian
#define DOG_BLOCK_LOG_DIMX 7
#define DOG_BLOCK_LOG_DIMY 0
#define DOG_BLOCK_DIMX (1 << DOG_BLOCK_LOG_DIMX)
#define DOG_BLOCK_DIMY (1 << DOG_BLOCK_LOG_DIMY)

//////////////////////////////////////////////////////////
//thread block size for keypoint detection
#define KEY_BLOCK_LOG_DIMX 3
#define KEY_BLOCK_LOG_DIMY 3
#define KEY_BLOCK_DIMX (1<<KEY_BLOCK_LOG_DIMX)
#define KEY_BLOCK_DIMY (1<<KEY_BLOCK_LOG_DIMY)
//#define KEY_OFFSET_ONE
//make KEY_BLOCK_LOG_DIMX 4 will make the write coalesced..
//but it seems uncoalesced writes don't affect the speed

//////////////////////////////////////////////////////////
//thread block size for initializing list generation (64, 128, 256, 512 ...) 实例化列表生成的线程块大小！
#define HIST_INIT_WIDTH 128
//thread block size for generating feature list (32, 64, 128, 256, 512, ...)生成特征列表的线程块大小
#define LISTGEN_BLOCK_DIM 128


/////////////////////////////////////////////////////////
//how many keypoint orientations to compute in a block
#define ORIENTATION_COMPUTE_PER_BLOCK 64
//how many keypoint descriptor to compute in a block (2, 4, 8, 16, 32)
#define DESCRIPTOR_COMPUTE_PER_BLOCK	4
#define DESCRIPTOR_COMPUTE_BLOCK_SIZE	(16 * DESCRIPTOR_COMPUTE_PER_BLOCK)
//how many keypoint descriptor to normalized in a block (32, ...)
#define DESCRIPTOR_NORMALIZ_PER_BLOCK	32



///////////////////////////////////////////
//Thread block size for visualization 
//(This doesn't affect the speed of computation)
#define BLOCK_LOG_DIM 4
#define BLOCK_DIM (1 << BLOCK_LOG_DIM)

//End SiftGPU setting section.
//----------------------------------------------------------------


__device__ __constant__ float d_kernel[KERNEL_MAX_WIDTH];
texture<float, 1, cudaReadModeElementType> texData;
texture<unsigned char, 1, cudaReadModeNormalizedFloat> texDataB;
texture<float2, 2, cudaReadModeElementType> texDataF2;
texture<float4, 1, cudaReadModeElementType> texDataF4;
texture<int4, 1, cudaReadModeElementType> texDataI4;
texture<int4, 1, cudaReadModeElementType> texDataList;

//template<int i>	 __device__ float Conv(float *data)		{    return Conv<i-1>(data) + data[i]*d_kernel[i];}
//template<>		__device__ float Conv<0>(float *data)	{    return data[0] * d_kernel[0];					}


//////////////////////////////////////////////////////////////
template<int FW> __global__ void FilterH( float* d_result, int width)
{

	const int HALF_WIDTH = FW >> 1; //    FW/2高斯卷积缩减一半
	const int CACHE_WIDTH = FILTERH_TILE_WIDTH + FW -1;  //128+FW-1     共享内存大小
	const int CACHE_COUNT = 2 + (CACHE_WIDTH - 2)/ FILTERH_TILE_WIDTH;   //一个catch包含几个高斯卷积核
	__shared__ float data[CACHE_WIDTH];  //128+一个高斯卷积核大小
	const int bcol = IMUL(blockIdx.x, FILTERH_TILE_WIDTH);  //128*blockIdx.x
	const int col =  bcol + threadIdx.x;  //线程索引
	const int index_min = IMUL(blockIdx.y, width);//每一行第一个
	const int index_max = index_min + width - 1;//每一行最后一个
	int src_index = index_min + bcol - HALF_WIDTH + threadIdx.x;  //每一行第blockIdx.x个线程块，减去高斯核的一半，有边缘效应！
	int cache_index = threadIdx.x; //0~128
	float value = 0;
#pragma unroll
	for(int j = 0; j < CACHE_COUNT; ++j)  //一个catch包含几个高斯卷积核
	{
		if(cache_index < CACHE_WIDTH)  //128+FW-128=FW  也就是前FW个线程运算了两次！！！
		{
			int fetch_index = src_index < index_min? index_min : (src_index > index_max ? index_max : src_index);
			data[cache_index] = tex1Dfetch(texData,fetch_index);
			src_index += FILTERH_TILE_WIDTH;
			cache_index += FILTERH_TILE_WIDTH;
		}
	}
	__syncthreads();  
	if(col >= width) return;
#pragma unroll
	for(int i = 0; i < FW; ++i)
	{
		value += (data[threadIdx.x + i]* d_kernel[i]);
	}
	//	value = Conv<FW-1>(data + threadIdx.x);
	d_result[index_min + col] = value;
}



////////////////////////////////////////////////////////////////////
template<int  FW>  __global__ void FilterV(float* d_result, int width, int height)
{
	const int HALF_WIDTH = FW >> 1;  //滤波的一半
	const int CACHE_WIDTH = FW + FILTERV_TILE_HEIGHT - 1;  //128+FW-1
	const int TEMP = CACHE_WIDTH & 0xf;//最大值是15
	//add some extra space to avoid bank conflict
#if FILTERV_TILE_WIDTH == 16
	//make the stride 16 * n +/- 1  步幅
	const int EXTRA = (TEMP == 1 || TEMP == 0) ? 1 - TEMP : 15 - TEMP;
#elif FILTERV_TILE_WIDTH == 8
	//make the stride 16 * n +/- 2
	const int EXTRA = (TEMP == 2 || TEMP == 1 || TEMP == 0) ? 2 - TEMP : (TEMP == 15? 3 : 14 - TEMP);
#elif FILTERV_TILE_WIDTH == 4
	//make the stride 16 * n +/- 4
	const int EXTRA = (TEMP >=0 && TEMP <=4) ? 4 - TEMP : (TEMP > 12? 20 - TEMP : 12 - TEMP);
#else
#error
#endif
	const int CACHE_TRUE_WIDTH = CACHE_WIDTH + EXTRA;//真正的共享内存宽度，
	const int CACHE_COUNT = (CACHE_WIDTH + FILTERV_BLOCK_HEIGHT - 1) / FILTERV_BLOCK_HEIGHT;  //catchlength/32
	const int WRITE_COUNT = (FILTERV_TILE_HEIGHT + FILTERV_BLOCK_HEIGHT -1) / FILTERV_BLOCK_HEIGHT;//128/32

	__shared__ float data[CACHE_TRUE_WIDTH * FILTERV_TILE_WIDTH];  //CACHE_TRUE_WIDTH*16
	const int row_block_first = IMUL(blockIdx.y, FILTERV_TILE_HEIGHT); //行索引的第一个值
	const int col = IMUL(blockIdx.x, FILTERV_TILE_WIDTH) + threadIdx.x; //这个才是正常的列索引
	const int row_first = row_block_first - HALF_WIDTH;//影像第一个行索引
	const int data_index_max = IMUL(height - 1, width) + col; //最后一行最后一个
	const int cache_col_start = threadIdx.y;	//列开始的地方
	const int cache_row_start = IMUL(threadIdx.x, CACHE_TRUE_WIDTH);
	int cache_index = cache_col_start + cache_row_start; //行列交叉
	int data_index = IMUL(row_first + cache_col_start, width) + col;//行列交叉

	if(col < width) 
	{
#pragma unroll
		for(int i = 0; i < CACHE_COUNT; ++i)
		{
			if(cache_col_start < CACHE_WIDTH - i * FILTERV_BLOCK_HEIGHT) 
			{
				int fetch_index = data_index < col ? col : (data_index > data_index_max? data_index_max : data_index);
				//如果把共享内存比作二位的，那么对应的就是左边一部分，在整个索引中分配是间断的！！！
				//随着循环的次数慢慢补齐你懂的
				data[cache_index + i * FILTERV_BLOCK_HEIGHT] = tex1Dfetch(texData,fetch_index);
				data_index += IMUL(FILTERV_BLOCK_HEIGHT, width);//每隔32行
			}
		}
	}
	__syncthreads();  //已完成共享内存的分配

	if(col >= width) return;

	int row = row_block_first + threadIdx.y;
	int index_start = cache_row_start + threadIdx.y;
#pragma unroll           //128/32                                             32                                                         32
	for(int i = 0; i < WRITE_COUNT;++i,row += FILTERV_BLOCK_HEIGHT, index_start += FILTERV_BLOCK_HEIGHT)
	{
		if(row < height)
		{
			int index_dest = IMUL(row, width) + col;
			float value = 0;
#pragma unroll
			for(int i = 0; i < FW; ++i)
			{
				value += (data[index_start + i] * d_kernel[i]);
			}
			d_result[index_dest] = value;
		}
	}
}


template<int LOG_SCALE> __global__ void UpsampleKernel(float* d_result, int width)
{
	const int SCALE = (1 << LOG_SCALE), SCALE_MASK = (SCALE - 1);
	const float INV_SCALE = 1.0f / (float(SCALE));
	int col = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;
	if(col >= width) return;

	int row = blockIdx.y >> LOG_SCALE; 
	int index = row * width + col;
	int dst_row = blockIdx.y;
	int dst_idx= (width * dst_row + col) * SCALE;
	int helper = blockIdx.y & SCALE_MASK; 
	if (helper)
	{
		float v11 = tex1Dfetch(texData, index);
		float v12 = tex1Dfetch(texData, index + 1);
		index += width;
		float v21 = tex1Dfetch(texData, index);
		float v22 = tex1Dfetch(texData, index + 1);
		float w1 = INV_SCALE * helper, w2 = 1.0 - w1;
		float v1 = (v21 * w1  + w2 * v11);
		float v2 = (v22 * w1  + w2 * v12);
		d_result[dst_idx] = v1;
#pragma unroll
		for(int i = 1; i < SCALE; ++i)
		{
			const float r2 = i * INV_SCALE;
			const float r1 = 1.0f - r2; 
			d_result[dst_idx +i] = v1 * r1 + v2 * r2;
		}
	}else
	{
		float v1 = tex1Dfetch(texData, index);
		float v2 = tex1Dfetch(texData, index + 1);
		d_result[dst_idx] = v1;
#pragma unroll
		for(int i = 1; i < SCALE; ++i)
		{
			const float r2 = i * INV_SCALE;
			const float r1 = 1.0f - r2; 
			d_result[dst_idx +i] = v1 * r1 + v2 * r2;
		}
	}

}

////////////////////////////////////////////////////////////////////////////////////////
void ProgramCU::SampleImageU(CuTexImage *dst, CuTexImage *src, int log_scale)
{
	int width = src->GetImgWidth(), height = src->GetImgHeight();
	src->BindTexture(texData);
	dim3 grid((width +  FILTERH_TILE_WIDTH - 1)/ FILTERH_TILE_WIDTH, height << log_scale);
	dim3 block(FILTERH_TILE_WIDTH);
	switch(log_scale)
	{
	case 1 : 	UpsampleKernel<1> <<< grid, block>>> ((float*) dst->_cuData, width);	break;
	case 2 : 	UpsampleKernel<2> <<< grid, block>>> ((float*) dst->_cuData, width);	break;
	case 3 : 	UpsampleKernel<3> <<< grid, block>>> ((float*) dst->_cuData, width);	break;
	default:	break;
	}
}

template<int LOG_SCALE> __global__ void DownsampleKernel(float* d_result, int src_width, int dst_width)
{
	const int dst_col = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;
	if(dst_col >= dst_width) return;
	const int src_col = min((dst_col << LOG_SCALE), (src_width - 1));  //dst_col*2
	const int dst_row = blockIdx.y;    //降采样影像
	const int src_row = blockIdx.y << LOG_SCALE;  //源影像  dst_row*2
	const int src_idx = IMUL(src_row, src_width) + src_col;
	const int dst_idx = IMUL(dst_width, dst_row) + dst_col;
	d_result[dst_idx] = tex1Dfetch(texData, src_idx);

}

__global__ void DownsampleKernel(float* d_result, int src_width, int dst_width, const int log_scale)
{
	const int dst_col = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;
	if(dst_col >= dst_width) return;
	const int src_col = min((dst_col << log_scale), (src_width - 1));
	const int dst_row = blockIdx.y; 
	const int src_row = blockIdx.y << log_scale;
	const int src_idx = IMUL(src_row, src_width) + src_col;
	const int dst_idx = IMUL(dst_width, dst_row) + dst_col;
	d_result[dst_idx] = tex1Dfetch(texData, src_idx);

}

void ProgramCU::SampleImageD(CuTexImage *dst, CuTexImage *src, int log_scale)
{
	int src_width = src->GetImgWidth(), dst_width = dst->GetImgWidth() ;

	src->BindTexture(texData);
	dim3 grid((dst_width +  FILTERH_TILE_WIDTH - 1)/ FILTERH_TILE_WIDTH, dst->GetImgHeight());
	dim3 block(FILTERH_TILE_WIDTH);
	switch(log_scale)
	{
	case 1 : 	DownsampleKernel<1> <<< grid, block>>> ((float*) dst->_cuData, src_width, dst_width);	break;
	case 2 :	DownsampleKernel<2> <<< grid, block>>> ((float*) dst->_cuData, src_width, dst_width);	break;
	case 3 : 	DownsampleKernel<3> <<< grid, block>>> ((float*) dst->_cuData, src_width, dst_width);	break;
	default:	DownsampleKernel    <<< grid, block>>> ((float*) dst->_cuData, src_width, dst_width, log_scale);
	}
		cudaThreadSynchronize();
}

__global__ void ChannelReduce_Kernel(float* d_result)
{
	int index = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;
	d_result[index] = tex1Dfetch(texData, index*4);
}

__global__ void ChannelReduce_Convert_Kernel(float* d_result)
{
	int index = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;
	float4 rgba = tex1Dfetch(texDataF4, index);
	d_result[index] = 0.299f * rgba.x + 0.587f* rgba.y + 0.114f * rgba.z;
}

void ProgramCU::ReduceToSingleChannel(CuTexImage* dst, CuTexImage* src, int convert_rgb)
{
	int width = src->GetImgWidth(), height = dst->GetImgHeight() ;

	dim3 grid((width * height +  FILTERH_TILE_WIDTH - 1)/ FILTERH_TILE_WIDTH);
	dim3 block(FILTERH_TILE_WIDTH);
	if(convert_rgb)
	{
		src->BindTexture(texDataF4);
		ChannelReduce_Convert_Kernel<<<grid, block>>>((float*)dst->_cuData);
	}else
	{
		src->BindTexture(texData);
		ChannelReduce_Kernel<<<grid, block>>>((float*)dst->_cuData);
	}
}

__global__ void ConvertByteToFloat_Kernel(float* d_result)
{
	int index = IMUL(blockIdx.x, FILTERH_TILE_WIDTH) + threadIdx.x;
	d_result[index] = tex1Dfetch(texDataB, index);
}

void ProgramCU::ConvertByteToFloat(CuTexImage*src, CuTexImage* dst)
{
	int width = src->GetImgWidth(), height = dst->GetImgHeight() ;
	dim3 grid((width * height +  FILTERH_TILE_WIDTH - 1)/ FILTERH_TILE_WIDTH);
	dim3 block(FILTERH_TILE_WIDTH);
	src->BindTexture(texDataB);
	ConvertByteToFloat_Kernel<<<grid, block>>>((float*)dst->_cuData);
}

void ProgramCU::CreateFilterKernel(float sigma, float* kernel, int& width)
{
	int i, sz = int( ceil( GlobalUtil::_FilterWidthFactor * sigma -0.5) ) ;//
	width = 2*sz + 1;

	if(width > KERNEL_MAX_WIDTH)
	{
		//filter size truncation
		sz = KERNEL_MAX_WIDTH >> 1;
		width =KERNEL_MAX_WIDTH;
	}else if(width < KERNEL_MIN_WIDTH)
	{
		sz = KERNEL_MIN_WIDTH >> 1;
		width =KERNEL_MIN_WIDTH;
	}

	float   rv = 1.0f/(sigma*sigma), v, ksum =0; 

	// pre-compute filter
	for( i = -sz ; i <= sz ; ++i) 
	{
		kernel[i+sz] =  v = exp(-0.5f * i * i *rv) ;
		ksum += v;
	}

	//normalize the kernel
	rv = 1.0f/ksum;
	for(i = 0; i< width ;i++) kernel[i]*=rv;
}


template<int FW> void ProgramCU::FilterImage(CuTexImage *dst, CuTexImage *src, CuTexImage* buf)
{
	int width = src->GetImgWidth(), height = src->GetImgHeight();






	//horizontal filtering
	src->BindTexture(texData); //src是源图像
	dim3 gridh((width +  FILTERH_TILE_WIDTH - 1)/ FILTERH_TILE_WIDTH, height);
	dim3 blockh(FILTERH_TILE_WIDTH);


			//GlobalUtil::StartTimer("水平");
	FilterH<FW><<<gridh, blockh>>>((float*)buf->_cuData, width);
	cudaThreadSynchronize();
	//	GlobalUtil::StopTimer();
	//	float _timing0 = GlobalUtil::GetElapsedTime();
	//CheckErrorCUDA("FilterH");




	///vertical filtering
	buf->BindTexture(texData);
	//16,128
	dim3 gridv((width + FILTERV_TILE_WIDTH - 1)/ FILTERV_TILE_WIDTH,  (height + FILTERV_TILE_HEIGHT - 1)/FILTERV_TILE_HEIGHT); //(50,5)
	dim3 blockv(FILTERV_TILE_WIDTH, FILTERV_BLOCK_HEIGHT);   //(16*32)
	//GlobalUtil::StartTimer("竖直");
	FilterV<FW><<<gridv, blockv>>>((float*)dst->_cuData, width, height); 
				cudaThreadSynchronize();			
		//		GlobalUtil::StopTimer();

		//float _timing1 = GlobalUtil::GetElapsedTime();
		
		
		//0.005,0.008   是原来的1.6倍！！！
	CheckErrorCUDA("FilterV");
}

//////////////////////////////////////////////////////////////////////
// tested on 2048x1500 image, the time on pyramid construction is
// OpenGL version : 18ms
// CUDA version: 28 ms
void ProgramCU::FilterImage(CuTexImage *dst, CuTexImage *src, CuTexImage* buf, float sigma)
{
	float filter_kernel[KERNEL_MAX_WIDTH]; int width;
	CreateFilterKernel(sigma, filter_kernel, width);
	cudaMemcpyToSymbol(d_kernel, filter_kernel, width * sizeof(float), 0, cudaMemcpyHostToDevice);

	switch(width)
	{
	case 5:		FilterImage< 5>(dst, src, buf);	break;
	case 7:		FilterImage< 7>(dst, src, buf);	break;
	case 9:		FilterImage< 9>(dst, src, buf);	break;
	case 11:	FilterImage<11>(dst, src, buf);	break;
	case 13:	FilterImage<13>(dst, src, buf);	break;
	case 15:	FilterImage<15>(dst, src, buf);	break;
	case 17:	FilterImage<17>(dst, src, buf);	break;
	case 19:	FilterImage<19>(dst, src, buf);	break;
	case 21:	FilterImage<21>(dst, src, buf);	break;
	case 23:	FilterImage<23>(dst, src, buf);	break;
	case 25:	FilterImage<25>(dst, src, buf);	break;
	case 27:	FilterImage<27>(dst, src, buf);	break;
	case 29:	FilterImage<29>(dst, src, buf);	break;
	case 31:	FilterImage<31>(dst, src, buf);	break;
	case 33:	FilterImage<33>(dst, src, buf);	break;
	default:	break;
	}

}


texture<float, 1, cudaReadModeElementType> texC;
texture<float, 1, cudaReadModeElementType> texP;
texture<float, 1, cudaReadModeElementType> texN;

void __global__ ComputeDOG_Kernel(float* d_dog, float2* d_got, int width, int height)
{
	int row = (blockIdx.y << DOG_BLOCK_LOG_DIMY) + threadIdx.y;
	int col = (blockIdx.x << DOG_BLOCK_LOG_DIMX) + threadIdx.x;
	if(col < width && row < height) 
	{
		int index = IMUL(row, width) + col;
		float vp = tex1Dfetch(texP, index);
		float v = tex1Dfetch(texC, index);
		d_dog[index] = v - vp;
		float vxn = tex1Dfetch(texC, index + 1);
		float vxp = tex1Dfetch(texC, index - 1);
		float vyp = tex1Dfetch(texC, index - width);
		float vyn = tex1Dfetch(texC, index + width);
		float dx = vxn - vxp, dy = vyn - vyp;
		float grd = 0.5f * sqrt(dx * dx  + dy * dy);
		float rot = (grd == 0.0f? 0.0f : atan2(dy, dx));
		d_got[index] = make_float2(grd, rot);
	}
}

void __global__ ComputeDOG_Kernel(float* d_dog, int width, int height)
{
	int row = (blockIdx.y << DOG_BLOCK_LOG_DIMY) + threadIdx.y;
	int col = (blockIdx.x << DOG_BLOCK_LOG_DIMX) + threadIdx.x;
	if(col < width && row < height) 
	{
		int index = IMUL(row, width) + col;
		float vp = tex1Dfetch(texP, index);
		float v = tex1Dfetch(texC, index);
		d_dog[index] = v - vp;
	}
}

void ProgramCU::ComputeDOG(CuTexImage* gus, CuTexImage* dog, CuTexImage* got)
{
	int width = gus->GetImgWidth(), height = gus->GetImgHeight();
	dim3 grid((width + DOG_BLOCK_DIMX - 1)/ DOG_BLOCK_DIMX,  (height + DOG_BLOCK_DIMY - 1)/DOG_BLOCK_DIMY);
	dim3 block(DOG_BLOCK_DIMX, DOG_BLOCK_DIMY);
	gus->BindTexture(texC);
	(gus -1)->BindTexture(texP);  //got实际上是前三层高斯金字塔的梯度值
	if(got->_cuData)
		ComputeDOG_Kernel<<<grid, block>>>((float*) dog->_cuData, (float2*) got->_cuData, width, height);
	else
		ComputeDOG_Kernel<<<grid, block>>>((float*) dog->_cuData, width, height);
	cudaThreadSynchronize();
}


#define READ_CMP_DOG_DATA(datai, tex, idx) \
	datai[0] = tex1Dfetch(tex, idx - 1);\
	datai[1] = tex1Dfetch(tex, idx);\
	datai[2] = tex1Dfetch(tex, idx + 1);\
	if(v > nmax)\
{\
	nmax = max(nmax, datai[0]);\
	nmax = max(nmax, datai[1]);\
	nmax = max(nmax, datai[2]);\
	if(v < nmax) goto key_finish;\
}else\
{\
	nmin = min(nmin, datai[0]);\
	nmin = min(nmin, datai[1]);\
	nmin = min(nmin, datai[2]);\
	if(v > nmin) goto key_finish;\
}


void __global__ ComputeKEY_Kernel(float4* d_key, int width, int colmax, int rowmax, 
	float dog_threshold0,  float dog_threshold, float edge_threshold, int subpixel_localization)
{
	float data[3][3], v;
	float datap[3][3], datan[3][3];
#ifdef KEY_OFFSET_ONE
	int row = (blockIdx.y << KEY_BLOCK_LOG_DIMY) + threadIdx.y + 1;
	int col = (blockIdx.x << KEY_BLOCK_LOG_DIMX) + threadIdx.x + 1;
#else
	int row = (blockIdx.y << KEY_BLOCK_LOG_DIMY) + threadIdx.y;
	int col = (blockIdx.x << KEY_BLOCK_LOG_DIMX) + threadIdx.x;
#endif
	int index = IMUL(row, width) + col;
	int idx[3] ={index - width, index, index + width};
	int in_image =0;
	float nmax, nmin, result = 0.0f;
	float dx = 0, dy = 0, ds = 0;
	bool offset_test_passed = true;
#ifdef KEY_OFFSET_ONE
	if(row < rowmax && col < colmax)
#else
	if(row > 0 && col > 0 && row < rowmax && col < colmax)
#endif
	{
		//一维抑制！！！！！！！
		in_image = 1;
		data[1][1] = v = tex1Dfetch(texC, idx[1]);  //当前像元
		if(fabs(v) <= dog_threshold0) goto key_finish;

		data[1][0] = tex1Dfetch(texC, idx[1] - 1);
		data[1][2] = tex1Dfetch(texC, idx[1] + 1);//左右像元
		nmax = max(data[1][0], data[1][2]);
		nmin = min(data[1][0], data[1][2]);//左右像元最大最小值

		if(v <=nmax && v >= nmin) goto key_finish; //极值是什么，是比所有值都大！！！当然比旁边的大，比旁边小的走开。
		//if((v > nmax && v < 0 )|| (v < nmin && v > 0)) goto key_finish;
		READ_CMP_DOG_DATA(data[0], texC, idx[0]);//第一行最大最小值
		READ_CMP_DOG_DATA(data[2], texC, idx[2]);//第三行最大最小值

		//二维抑制！！！！
		//edge supression  边缘检测！
		float vx2 = v * 2.0f;
		float fxx = data[1][0] + data[1][2] - vx2;
		float fyy = data[0][1] + data[2][1] - vx2;
		float fxy = 0.25f * (data[2][2] + data[0][0] - data[2][0] - data[0][2]);
		float temp1 = fxx * fyy - fxy * fxy;
		float temp2 = (fxx + fyy) * (fxx + fyy);
		if(temp1 <=0 || temp2 > edge_threshold * temp1) goto key_finish;
		
		//三维抑制！！！！
		//read the previous level
		READ_CMP_DOG_DATA(datap[0], texP, idx[0]);//上一层的最大最小值
		READ_CMP_DOG_DATA(datap[1], texP, idx[1]);
		READ_CMP_DOG_DATA(datap[2], texP, idx[2]);


		//read the next level
		READ_CMP_DOG_DATA(datan[0], texN, idx[0]);//下一层的最大最小值
		READ_CMP_DOG_DATA(datan[1], texN, idx[1]);
		READ_CMP_DOG_DATA(datan[2], texN, idx[2]);

		if(subpixel_localization)
		{
			//subpixel localization
			float fx = 0.5f * (data[1][2] - data[1][0]);
			float fy = 0.5f * (data[2][1] - data[0][1]);
			float fs = 0.5f * (datan[1][1] - datap[1][1]);

			float fss = (datan[1][1] + datap[1][1] - vx2);
			float fxs = 0.25f* (datan[1][2] + datap[1][0] - datan[1][0] - datap[1][2]);
			float fys = 0.25f* (datan[2][1] + datap[0][1] - datan[0][1] - datap[2][1]);

			//need to solve dx, dy, ds;
			// |-fx|     | fxx fxy fxs |   |dx|
			// |-fy|  =  | fxy fyy fys | * |dy|
			// |-fs|     | fxs fys fss |   |ds|
			float4 A0 = fxx > 0? make_float4(fxx, fxy, fxs, -fx) : make_float4(-fxx, -fxy, -fxs, fx);
			float4 A1 = fxy > 0? make_float4(fxy, fyy, fys, -fy) : make_float4(-fxy, -fyy, -fys, fy);
			float4 A2 = fxs > 0? make_float4(fxs, fys, fss, -fs) : make_float4(-fxs, -fys, -fss, fs);
			//高斯消元法
			float maxa = max(max(A0.x, A1.x), A2.x);  //选主元
			if(maxa >= 1e-10)
			{
				if(maxa == A1.x)//如果A1是最大的，那么A1和A0调换
				{
					float4 TEMP = A1; A1 = A0; A0 = TEMP;  
				}else if(maxa == A2.x)//如果A2是最大的，那么A1和A0调换
				{
					float4 TEMP = A2; A2 = A0; A0 = TEMP;
				}
				A0.y /= A0.x;	A0.z /= A0.x;	A0.w/= A0.x;
				A1.y -= A1.x * A0.y;	A1.z -= A1.x * A0.z;	A1.w -= A1.x * A0.w;
				A2.y -= A2.x * A0.y;	A2.z -= A2.x * A0.z;	A2.w -= A2.x * A0.w;
				if(abs(A2.y) > abs(A1.y))
				{
					float4 TEMP = A2;	A2 = A1; A1 = TEMP;
				}
				if(abs(A1.y) >= 1e-10) 
				{
					A1.z /= A1.y;	A1.w /= A1.y;
					A2.z -= A2.y * A1.z;	A2.w -= A2.y * A1.w;
					if(abs(A2.z) >= 1e-10) 
					{
						ds = A2.w / A2.z;
						dy = A1.w - ds * A1.z;
						dx = A0.w - ds * A0.z - dy * A0.y;

						offset_test_passed = 
							fabs(data[1][1] + 0.5f * (dx * fx + dy * fy + ds * fs)) > dog_threshold   //去除低对比度的点
							&&fabs(ds) < 1.0f && fabs(dx) < 1.0f && fabs(dy) < 1.0f;
					}
				}
			}
		}
		if(offset_test_passed) result = v > nmax ? 1.0 : -1.0;//当前像元通过测试，大于最大值为极大值，否则为极小值！
	}
key_finish:  //已经知道位置了index就代表了行列，但是保存不了那么多信息
		if(in_image) d_key[index] = make_float4(result, dx, dy, ds);  //得到像元的改正值！(用周围像元替代该像元直到收敛，应该是以后的事）
}


void ProgramCU::ComputeKEY(CuTexImage* dog, CuTexImage* key, float Tdog, float Tedge)
{
	int width = dog->GetImgWidth(), height = dog->GetImgHeight();
	float Tdog1 = (GlobalUtil::_SubpixelLocalization? 0.8f : 1.0f) * Tdog;
	CuTexImage* dogp = dog - 1;
	CuTexImage* dogn = dog + 1;
#ifdef KEY_OFFSET_ONE
	dim3 grid((width - 1 + KEY_BLOCK_DIMX - 1)/ KEY_BLOCK_DIMX,  (height - 1 + KEY_BLOCK_DIMY - 1)/KEY_BLOCK_DIMY);
#else
	dim3 grid((width + KEY_BLOCK_DIMX - 1)/ KEY_BLOCK_DIMX,  (height + KEY_BLOCK_DIMY - 1)/KEY_BLOCK_DIMY);
#endif
	dim3 block(KEY_BLOCK_DIMX, KEY_BLOCK_DIMY);
	dogp->BindTexture(texP);
	dog ->BindTexture(texC);
	dogn->BindTexture(texN);
	Tedge = (Tedge+1)*(Tedge+1)/Tedge;
	//(8,8) (800/8,600/8)
	ComputeKEY_Kernel<<<grid, block>>>((float4*) key->_cuData, width,
		width -1, height -1, Tdog1, Tdog, Tedge, GlobalUtil::_SubpixelLocalization);
		cudaThreadSynchronize();
}


//ws 800,wd 200,height 600
void __global__ InitHist_Kernel(int4* hist, int ws, int wd, int height)
{
	int row = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;//行索引
	int col = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;//列索引
	if(row < height && col < wd)
	{
		int hidx = IMUL(row, wd) + col; //直方图索引(0,1,2,3,4...200)
		int scol = col << 2;//乘以4  (0,4,8,12...800)
		int sidx = IMUL(row, ws) + scol;//影像索引(0,4,8,12...800)
		int v[4] = {0, 0, 0, 0}; 
		if(row > 0 && row < height -1)
		{
#pragma unroll
			for(int i = 0; i < 4 ; ++i, ++scol)
			{
				float4 temp = tex1Dfetch(texDataF4, sidx +i);//当前像元以及右边三个
				//temp(result, dx, dy, ds)，result不为0说明是特征点！
				v[i] = (scol < ws -1 && scol > 0 && temp.x!=0) ? 1 : 0;//满足条件，不超过列索引，且temp.x不为零       则为1
			}
		}
		hist[hidx] = make_int4(v[0], v[1], v[2], v[3]);//高度不变，宽度变为之前的1/4

	}
}



void ProgramCU::InitHistogram(CuTexImage* key, CuTexImage* hist)
{
	int ws = key->GetImgWidth(), hs = key->GetImgHeight();//800*600
	int wd = hist->GetImgWidth(), hd = hist->GetImgHeight();//200*600
	dim3 grid((wd  + HIST_INIT_WIDTH - 1)/ HIST_INIT_WIDTH,  hd);//(200/128,600)
	dim3 block(HIST_INIT_WIDTH, 1);  //(128,1)
	key->BindTexture(texDataF4);
	//hist->cuda,800,200,600
	InitHist_Kernel<<<grid, block>>>((int4*) hist->_cuData, ws, wd, hd);
		cudaThreadSynchronize();
}


//200,50,600
void __global__ ReduceHist_Kernel(int4* d_hist, int ws, int wd, int height)
{
	int row = IMUL(blockIdx.y, blockDim.y) + threadIdx.y;//行索引
	int col = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;//列索引
	if(row < height && col < wd)
	{
		int hidx = IMUL(row, wd) + col;//直方图索引
		int scol = col << 2;//col*4
		int sidx = IMUL(row, ws) + scol;//上一层直方图索引
		int v[4] = {0, 0, 0, 0}; 
#pragma unroll
		for(int i = 0; i < 4 && scol < ws; ++i, ++scol)
		{
			int4 temp = tex1Dfetch(texDataI4, sidx + i);//上一层直方图中的像元和右边三个像元  200--50四倍关系！
			v[i] = temp.x + temp.y + temp.z + temp.w; //直方图的四个元素(x,y,z,w)
		}
		d_hist[hidx] = make_int4(v[0], v[1], v[2], v[3]);//本层的直方图索引
	}
}

void ProgramCU::ReduceHistogram(CuTexImage*hist1, CuTexImage* hist2)
{
	int ws = hist1->GetImgWidth(), hs = hist1->GetImgHeight();
	int wd = hist2->GetImgWidth(), hd = hist2->GetImgHeight();
	int temp = (int)floor(logf(float(wd * 2/ 3)) / logf(2.0f));
	const int wi = min(7, max(temp , 0));
	hist1->BindTexture(texDataI4);

	const int BW = 1 << wi, BH =  1 << (7 - wi);
	dim3 grid((wd  + BW - 1)/ BW,  (hd + BH -1) / BH);//（wd/32,hd/4）
	dim3 block(BW, BH);//（32,4）
	ReduceHist_Kernel<<<grid, block>>>((int4*)hist2->_cuData, ws, wd, hd);
		cudaThreadSynchronize();
}


void __global__ ListGen_Kernel(int4* d_list, int width)
{
	int idx1 = IMUL(blockIdx.x, blockDim.x) + threadIdx.x; //特征点索引
	int4 pos = tex1Dfetch(texDataList, idx1);  //拾取纹理内存
	int idx2 = IMUL(pos.y, width) + pos.x; //直方图索引（不为0的）   y指600行中的第几行，即第几个特征点，x为特征点具体位置，第一次width为4
	int4 temp = tex1Dfetch(texDataI4, idx2);//拾取直方图纹理内存
	int  sum1 = temp.x + temp.y;
	int  sum2 = sum1 + temp.z;                                                                                                                                                             
	pos.x <<= 2;//pos.x *4
	if(pos.z >= sum2)//这个设计的好巧妙呀！！！  先算pos.x=0的，再把pos.x不等于0的化为0，再按照pos.x=0的情况算
	{
		pos.x += 3;
		pos.z -= sum2;
	}else if(pos.z >= sum1)
	{
		pos.x += 2;
		pos.z -= sum1;
	}else if(pos.z >= temp.x)
	{
		pos.x += 1;
		pos.z -= temp.x;
	}
	d_list[idx1] = pos;
}

//input list (x, y) (x, y) ....           特征feature层，hist层直方图层
void ProgramCU::GenerateList(CuTexImage* list, CuTexImage* hist)
{
	int len = list->GetImgWidth();//327个特征点
	list->BindTexture(texDataList); 
	hist->BindTexture(texDataI4);
	dim3  grid((len + LISTGEN_BLOCK_DIM -1) /LISTGEN_BLOCK_DIM);//  len/128
	dim3  block(LISTGEN_BLOCK_DIM);//128
	//listgenerate，列表生成核函数
	ListGen_Kernel<<<grid, block>>>((int4*) list->_cuData, hist->GetImgWidth());
		cudaThreadSynchronize();
}

void __global__ ComputeOrientation_Kernel(float4* d_list, 
	int list_len,//327
	int width, int height,  //800,600
	float sigma, float sigma_step, //2.01,1.26
	float gaussian_factor, float sample_factor,//1.5,3
	int num_orientation,//2
	int existing_keypoint, //0
	int subpixel, //1
	int keepsign)//0
{
	//10度每半径
	const float ten_degree_per_radius = 5.7295779513082320876798154814105; //(360/10)/2pi
	//半径每十度
	const float radius_per_ten_degrees = 1.0 / 5.7295779513082320876798154814105;//(10/360)*2PI
	int idx = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;//线程索引
	if(idx >= list_len) return;
	float4 key; //重构key！！！(x,y,sigma,)
	if(existing_keypoint)
	{
		key = tex1Dfetch(texDataF4, idx);		
	}else 
	{
		int4 ikey = tex1Dfetch(texDataList, idx);//(x,y,sigma,)
		key.x = ikey.x + 0.5f; //四舍五入
		key.y = ikey.y + 0.5f;
		key.z = sigma;
		if(subpixel || keepsign)
		{
			float4 offset = tex1Dfetch(texDataF4, IMUL(width, ikey.y) + ikey.x);//定位到关键点
			if(subpixel)
			{
				//key（result,dx,dy,ds）
				key.x += offset.y;//x+=dx
				key.y += offset.z;//y+=dy
				key.z *= pow(sigma_step, offset.w);//z*=dz
			}
			if(keepsign) key.z *= offset.x; //???
		}
	}
	if(num_orientation == 0)
	{
		key.w = 0;
		d_list[idx] = key;
		return;
	}
	float vote[37]; //权！！！
	float gsigma = key.z * gaussian_factor;//key.z就是sigma
	float win = fabs(key.z) * sample_factor;//窗口
	float dist_threshold = win * win + 0.5;  //距离阈值
	float factor = -0.5f / (gsigma * gsigma);
	float xmin = max(1.5f, floor(key.x - win) + 0.5f);  //-radius
	float ymin = max(1.5f, floor(key.y - win) + 0.5f);
	float xmax = min(width - 1.5f, floor(key.x + win) + 0.5f);//+radius
	float ymax = min(height -1.5f, floor(key.y + win) + 0.5f);
#pragma unroll
	for(int i = 0; i < 36; ++i) vote[i] = 0.0f;  //36个方向
	for(float y = ymin; y <= ymax; y += 1.0f) //-radius~+radius
	{
		for(float x = xmin; x <= xmax; x += 1.0f)//-radius~+radius
		{
			float dx = x - key.x;  //i
			float dy = y - key.y; //j
			float sq_dist  = dx * dx + dy * dy;

			float2 got = tex2D(texDataF2, x, y);// 二维纹理  got.x是梯度幅值，got.y是梯度方向
			float weight = got.x * exp(sq_dist * factor);//该点相对于特征点的高斯权重！！！   这个是梯度的幅值
			float fidx = floor(got.y * ten_degree_per_radius);//不超过2PI的乘以5肯定不超过36！
			int oidx = fidx;
			if(oidx < 0) oidx += 36;
			vote[oidx] += weight; //可能有很多个的！！！累加到这个方向
		}
	}

	//filter the vote   高斯权滤波

	const float one_third = 1.0 /3.0;//三分之一
#pragma unroll  //循环展开
	for(int i = 0; i < 6; ++i)
	{
		vote[36] = vote[0];
		float pre = vote[35];
#pragma unroll
		for(int j = 0; j < 36; ++j)
		{
			float temp = one_third * (pre + vote[j] + vote[j + 1]);
			pre = vote[j];			vote[j] = temp;
			//1、当前值赋值给pre2、当前值为左边值和右边值中间值平均值
		}
	}

	vote[36] = vote[0];
	if(num_orientation == 1 || existing_keypoint)
	{
		int index_max = 0;
		float max_vote = vote[0];
#pragma unroll
		for(int i = 1; i < 36; ++i)
		{
			index_max =  vote[i] > max_vote? i : index_max;
			max_vote = max(max_vote, vote[i]);
		}
		float pre = vote[index_max == 0? 35 : index_max -1];
		float next = vote[index_max + 1];
		float weight = max_vote;
		float off =  0.5f * FDIV(next - pre, weight + weight - next - pre);
		key.w = radius_per_ten_degrees * (index_max + 0.5f + off);
		d_list[idx] = key;

	}else
	{
		float max_vote = vote[0];
#pragma unroll
		for(int i = 1; i < 36; ++i)		max_vote = max(max_vote, vote[i]);  //找到最大权值

		float vote_threshold = max_vote * 0.8f;  //80% 峰值！！
		float pre = vote[35];
		float max_rot[2], max_vot[2] = {0, 0};  //主方向和辅方向
		int  ocount = 0;
#pragma unroll
		for(int i =0; i < 36; ++i)//36个方向
		{
			float next = vote[i + 1];//下一个
			if(vote[i] > vote_threshold && vote[i] > pre && vote[i] > next)
			{
				float di = 0.5f * FDIV(next - pre, vote[i] + vote[i] - next - pre);//除以
				float rot = i + di + 0.5f;
				float weight = vote[i];
				///得到的结果是max_vot[0]是最大的    max_vot[1]>=max_vot[0]
				if(weight > max_vot[1])
				{
					if(weight > max_vot[0])
					{
						max_vot[1] = max_vot[0]; //
						max_rot[1] = max_rot[0];
						max_vot[0] = weight;
						max_rot[0] = rot;
					}
					else
					{
						max_vot[1] = weight;
						max_rot[1] = rot;
					}
					ocount ++;
				}
			}
			pre = vote[i];
		}
		float fr1 = max_rot[0] / 36.0f; //归一化（0,1）主方向  得出的是一个百分比！
		if(fr1 < 0) fr1 += 1.0f; 
		unsigned short us1 = ocount == 0? 65535 : ((unsigned short )floor(fr1 * 65535.0f));
		unsigned short us2 = 65535; 
		if(ocount > 1)  //如果ocount为1，则说明没有辅助方向，大于1则说明有辅方向
		{
			float fr2 = max_rot[1] / 36.0f; //归一化（0,1）
			if(fr2 < 0) fr2 += 1.0f;
			us2 = (unsigned short ) floor(fr2 * 65535.0f);
		}
		unsigned int uspack = (us2 << 16) | us1; //us2*2^16   us2移位运算!!! 保留了两个方向！！！
		//数学不好是坑啊，3二进制是11,3*2=6二进制是110，相当于每乘以2后面加一个0
		//也就是加了16个0，也就是避开了65535！保留了主方向和辅助方向！
		key.w = __int_as_float(uspack);//把int作为float保存在key.w中，别忘了key的数据类型是！！！float4
		d_list[idx] = key;
	}

}




void ProgramCU::ComputeOrientation(CuTexImage* list, CuTexImage* got, CuTexImage*key, 
	float sigma, float sigma_step, int existing_keypoint)
{
	int len = list->GetImgWidth();
	if(len <= 0) return;
	int width = got->GetImgWidth(), height = got->GetImgHeight();
	if(existing_keypoint)
	{
		list->BindTexture(texDataF4);
	}else
	{
		list->BindTexture(texDataList);
		if(GlobalUtil::_SubpixelLocalization) key->BindTexture(texDataF4);//F4(result,dx,dy,ds)
	}
	got->BindTexture2D(texDataF2);  //F2,梯度和角度

	const int block_width = len < ORIENTATION_COMPUTE_PER_BLOCK ? 16 : ORIENTATION_COMPUTE_PER_BLOCK;   //len<64,以16位大小，大于以64
	dim3 grid((len + block_width -1) / block_width);
	dim3 block(block_width);

	ComputeOrientation_Kernel<<<grid, block>>>((float4*) list->_cuData, 
		len, width, height, sigma, sigma_step, 
		GlobalUtil::_OrientationGaussianFactor, 
		GlobalUtil::_OrientationGaussianFactor * GlobalUtil::_OrientationWindowFactor,
		GlobalUtil::_FixedOrientation? 0 : GlobalUtil::_MaxOrientation, //0是假
		existing_keypoint, GlobalUtil::_SubpixelLocalization, GlobalUtil::_KeepExtremumSign);
		cudaThreadSynchronize();

	ProgramCU::CheckErrorCUDA("ComputeOrientation");
}

template <bool DYNAMIC_INDEXING> void __global__ ComputeDescriptor_Kernel(float4* d_des, int num, 
	int width, int height, float window_factor)
{
	const float rpi = 4.0/ 3.14159265358979323846;
	int idx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	int fidx = idx >> 4;  //idx/16 得到真正的特征点个数
	if(fidx >= num) return;
	float4 key = tex1Dfetch(texDataF4, fidx);//featureTex特征点定位
	int bidx = idx& 0xf, ix = bidx & 0x3, iy = bidx >> 2;
	//(0,0),(0,1)(0,2)...(3,3)
	float spt = fabs(key.z * window_factor);//窗口大小  3sigma
	float s, c; __sincosf(key.w, &s, &c);  //旋转到主方向！！！
	float anglef = key.w > 3.14159265358979323846? key.w - (2.0 * 3.14159265358979323846) : key.w ; 
	float cspt = c * spt, sspt = s * spt;//单独看并没有什么意义
	float crspt = c / spt, srspt = s / spt;//单独看并没有什么意义
	float2 offsetpt, pt;
	float xmin, ymin, xmax, ymax, bsz;
	offsetpt.x = ix - 1.5f;//哦~~这个1.5好熟悉呀，原来是那个+d/2-0.5
	offsetpt.y = iy - 1.5f;//将(0~3)->转变为(-1.5~1.5)
	//种子点的坐标
	pt.x = cspt * offsetpt.x - sspt * offsetpt.y + key.x;  //关键点旋转后的坐标？
	pt.y = cspt * offsetpt.y + sspt * offsetpt.x + key.y;
	bsz =  fabs(cspt) + fabs(sspt); //radius  √2*spt
	xmin = max(1.5f, floor(pt.x - bsz) + 0.5f); //-radius~radius
	ymin = max(1.5f, floor(pt.y - bsz) + 0.5f); //-radius~radius
	xmax = min(width - 1.5f, floor(pt.x + bsz) + 0.5f);
	ymax = min(height - 1.5f, floor(pt.y + bsz) + 0.5f);
	float des[9];
#pragma unroll
	for(int i =0; i < 9; ++i) des[i] = 0.0f;
	//前面不管，已经确定了邻域窗口大小了
	for(float y = ymin; y <= ymax; y += 1.0f)
	{
		for(float x = xmin; x <= xmax; x += 1.0f)
		{
			float dx = x - pt.x;   //pt为中心点坐标
			float dy = y - pt.y;
			float nx = crspt * dx + srspt * dy; //得到区域坐标
			float ny = crspt * dy - srspt * dx;
			float nxn = fabs(nx);
			float nyn = fabs(ny);
			if(nxn < 1.0f && nyn < 1.0f)
			{
				float2 cc = tex2D(texDataF2, x, y);  //梯度金字塔
				float dnx = nx + offsetpt.x;
				float dny = ny + offsetpt.y;//又旋转回来了！！！
				float ww = exp(-0.125f * (dnx * dnx + dny * dny));
				float wx = 1.0 - nxn;
				float wy = 1.0 - nyn;
				float weight = ww * wx * wy * cc.x;
				float theta = (anglef - cc.y) * rpi; //旋转到主方向
				if(theta < 0) theta += 8.0f;
				float fo = floor(theta);
				int fidx = fo;//方向
				float weight1 = fo + 1.0f  - theta;  //1-(theta - fo)
				float weight2 = theta - fo;   //theta - fo
				if(DYNAMIC_INDEXING)
				{
					des[fidx] += (weight1 * weight);
					des[fidx + 1] += (weight2 * weight);
					//this dynamic indexing part might be slow
				}else
				{
#pragma unroll
					for(int k = 0; k < 8; ++k)
					{
						if(k == fidx) 
						{
							des[k] += (weight1 * weight);
							des[k+1] += (weight2 * weight);
						}
					}
				}
			}
		}
	}
	des[0] += des[8];

	int didx = idx << 1;//0,2,4,6,8
	//每个特征点八个方向，16个特征区域，16*8=128
	//32个idx为一个特征点！！！！
	//128(float)*num->32(float4)*num->2(float4)*  16*num



	//16个线程，16<<1=16*2  每个线程计算8个方向
	d_des[didx] = make_float4(des[0], des[1], des[2], des[3]);
	d_des[didx+1] = make_float4(des[4], des[5], des[6], des[7]);
}


template <bool DYNAMIC_INDEXING> void __global__ ComputeDescriptorRECT_Kernel(float4* d_des, int num, 
	int width, int height, float window_factor)
{
	const float rpi = 4.0/ 3.14159265358979323846;
	int idx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	int fidx = idx >> 4;
	if(fidx >= num) return;
	float4 key = tex1Dfetch(texDataF4, fidx);
	int bidx = idx& 0xf, ix = bidx & 0x3, iy = bidx >> 2;
	//float aspect_ratio = key.w / key.z;
	//float aspect_sq = aspect_ratio * aspect_ratio;
	float sptx = key.z * 0.25, spty = key.w * 0.25;
	float xmin, ymin, xmax, ymax; float2 pt;
	pt.x = sptx * (ix + 0.5f)  + key.x;
	pt.y = spty * (iy + 0.5f)  + key.y;
	xmin = max(1.5f, floor(pt.x - sptx) + 0.5f);
	ymin = max(1.5f, floor(pt.y - spty) + 0.5f);
	xmax = min(width - 1.5f, floor(pt.x + sptx) + 0.5f);
	ymax = min(height - 1.5f, floor(pt.y + spty) + 0.5f);
	float des[9];
#pragma unroll
	for(int i =0; i < 9; ++i) des[i] = 0.0f;
	for(float y = ymin; y <= ymax; y += 1.0f)
	{
		for(float x = xmin; x <= xmax; x += 1.0f)
		{
			float nx = (x - pt.x) / sptx;
			float ny = (y - pt.y) / spty;
			float nxn = fabs(nx);
			float nyn = fabs(ny);
			if(nxn < 1.0f && nyn < 1.0f)
			{
				float2 cc = tex2D(texDataF2, x, y);
				float wx = 1.0 - nxn;
				float wy = 1.0 - nyn;
				float weight =  wx * wy * cc.x;
				float theta = (- cc.y) * rpi;
				if(theta < 0) theta += 8.0f;
				float fo = floor(theta);
				int fidx = fo;
				float weight1 = fo + 1.0f  - theta;
				float weight2 = theta - fo;
				if(DYNAMIC_INDEXING)
				{
					des[fidx] += (weight1 * weight);
					des[fidx + 1] += (weight2 * weight);
					//this dynamic indexing part might be slow
				}else
				{
#pragma unroll
					for(int k = 0; k < 8; ++k)
					{
						if(k == fidx) 
						{
							des[k] += (weight1 * weight);
							des[k+1] += (weight2 * weight);
						}
					}
				}
			}
		}
	}
	des[0] += des[8];

	int didx = idx << 1;
	d_des[didx] = make_float4(des[0], des[1], des[2], des[3]);
	d_des[didx+1] = make_float4(des[4], des[5], des[6], des[7]);
}

void __global__ NormalizeDescriptor_Kernel(float4* d_des, int num)
{
	float4 temp[32];
	int idx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	if(idx >= num) return;
	int sidx = idx << 5;//idx*32
	float norm1 = 0, norm2 = 0;
#pragma unroll
	for(int i = 0; i < 32; ++i)  //32*4=128
	{
		temp[i] = tex1Dfetch(texDataF4, sidx +i);
		norm1 += (temp[i].x * temp[i].x + temp[i].y * temp[i].y +
			temp[i].z * temp[i].z + temp[i].w * temp[i].w);
	}
	norm1 = rsqrt(norm1);  //分母

#pragma unroll
	for(int i = 0; i < 32; ++i)
	{
		temp[i].x = min(0.2f, temp[i].x * norm1);
		temp[i].y = min(0.2f, temp[i].y * norm1);
		temp[i].z = min(0.2f, temp[i].z * norm1);
		temp[i].w = min(0.2f, temp[i].w * norm1);
		norm2 += (temp[i].x * temp[i].x + temp[i].y * temp[i].y +
			temp[i].z * temp[i].z + temp[i].w * temp[i].w);//
	}

	norm2 = rsqrt(norm2);
#pragma unroll
	for(int i = 0; i < 32; ++i)
	{
		temp[i].x *= norm2;		temp[i].y *= norm2;
		temp[i].z *= norm2;		temp[i].w *= norm2;
		d_des[sidx + i] = temp[i];
	}
}

void ProgramCU::ComputeDescriptor(CuTexImage*list, CuTexImage* got, CuTexImage* dtex, int rect, int stream)
{
	int num = list->GetImgWidth();
	int width = got->GetImgWidth();
	int height = got->GetImgHeight();

	dtex->InitTexture(num * 128, 1, 1);
	got->BindTexture2D(texDataF2);
	list->BindTexture(texDataF4);
	int block_width = DESCRIPTOR_COMPUTE_BLOCK_SIZE;//64
	dim3 grid((num * 16 + block_width -1) / block_width);//num*16/64    16*8=128
	dim3 block(block_width);//64

	if(rect)
	{
		if(GlobalUtil::_UseDynamicIndexing)
			ComputeDescriptorRECT_Kernel<true><<<grid, block>>>((float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
		else
			ComputeDescriptorRECT_Kernel<false><<<grid, block>>>((float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);

	}else
	{
		if(GlobalUtil::_UseDynamicIndexing)
			ComputeDescriptor_Kernel<true><<<grid, block>>>((float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
		else
			ComputeDescriptor_Kernel<false><<<grid, block>>>((float4*) dtex->_cuData, num, width, height, GlobalUtil::_DescriptorWindowFactor);
	}
			cudaThreadSynchronize();
	if(GlobalUtil::_NormalizedSIFT)
	{
		dtex->BindTexture(texDataF4);
		const int block_width = DESCRIPTOR_NORMALIZ_PER_BLOCK;
		dim3 grid((num + block_width -1) / block_width);
		dim3 block(block_width);
		NormalizeDescriptor_Kernel<<<grid, block>>>((float4*) dtex->_cuData, num);
				cudaThreadSynchronize();
	}
	CheckErrorCUDA("ComputeDescriptor");
}

//////////////////////////////////////////////////////
void ProgramCU::FinishCUDA()
{
	cudaThreadSynchronize();
}

int ProgramCU::CheckErrorCUDA(const char* location)
{
	cudaError_t e = cudaGetLastError();
	if(e)
	{
		if(location) fprintf(stderr, "%s:\t",  location);
		fprintf(stderr, "%s\n",  cudaGetErrorString(e));
		//assert(0);
		return 1;
	}else
	{
		return 0; 
	}
}

void __global__ ConvertDOG_Kernel(float* d_result, int width, int height)
{
	int row = (blockIdx.y << BLOCK_LOG_DIM) + threadIdx.y;
	int col = (blockIdx.x << BLOCK_LOG_DIM) + threadIdx.x;
	if(col < width && row < height)
	{
		int index = row * width  + col;
		float v = tex1Dfetch(texData, index);
		d_result[index] = (col == 0 || row == 0 || col == width -1 || row == height -1)?
			0.5 : saturate(0.5+20.0*v);
	}
}
///
void ProgramCU::DisplayConvertDOG(CuTexImage* dog, CuTexImage* out)
{
	if(out->_cuData == NULL) return;
	int width = dog->GetImgWidth(), height = dog ->GetImgHeight();
	dog->BindTexture(texData);
	dim3 grid((width + BLOCK_DIM - 1)/ BLOCK_DIM,  (height + BLOCK_DIM - 1)/BLOCK_DIM);
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	ConvertDOG_Kernel<<<grid, block>>>((float*) out->_cuData, width, height);
	ProgramCU::CheckErrorCUDA("DisplayConvertDOG");
}

void __global__ ConvertGRD_Kernel(float* d_result, int width, int height)
{
	int row = (blockIdx.y << BLOCK_LOG_DIM) + threadIdx.y;
	int col = (blockIdx.x << BLOCK_LOG_DIM) + threadIdx.x;
	if(col < width && row < height)
	{
		int index = row * width  + col;
		float v = tex1Dfetch(texData, index << 1);
		d_result[index] = (col == 0 || row == 0 || col == width -1 || row == height -1)?
			0 : saturate(5 * v);

	}
}


void ProgramCU::DisplayConvertGRD(CuTexImage* got, CuTexImage* out)
{
	if(out->_cuData == NULL) return;
	int width = got->GetImgWidth(), height = got ->GetImgHeight();
	got->BindTexture(texData);
	dim3 grid((width + BLOCK_DIM - 1)/ BLOCK_DIM,  (height + BLOCK_DIM - 1)/BLOCK_DIM);
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	ConvertGRD_Kernel<<<grid, block>>>((float*) out->_cuData, width, height);
	ProgramCU::CheckErrorCUDA("DisplayConvertGRD");
}

void __global__ ConvertKEY_Kernel(float4* d_result, int width, int height)
{

	int row = (blockIdx.y << BLOCK_LOG_DIM) + threadIdx.y;
	int col = (blockIdx.x << BLOCK_LOG_DIM) + threadIdx.x;
	if(col < width && row < height)
	{
		int index = row * width + col;
		float4 keyv = tex1Dfetch(texDataF4, index);
		int is_key = (keyv.x == 1.0f || keyv.x == -1.0f); 
		int inside = col > 0 && row > 0 && row < height -1 && col < width - 1;
		float v = inside? saturate(0.5 + 20 * tex1Dfetch(texData, index)) : 0.5;
		d_result[index] = is_key && inside ? 
			(keyv.x > 0? make_float4(1.0f, 0, 0, 1.0f) : make_float4(0.0f, 1.0f, 0.0f, 1.0f)): 
			make_float4(v, v, v, 1.0f) ;
	}
}
void ProgramCU::DisplayConvertKEY(CuTexImage* key, CuTexImage* dog, CuTexImage* out)
{
	if(out->_cuData == NULL) return;
	int width = key->GetImgWidth(), height = key ->GetImgHeight();
	dog->BindTexture(texData);
	key->BindTexture(texDataF4);
	dim3 grid((width + BLOCK_DIM - 1)/ BLOCK_DIM,  (height + BLOCK_DIM - 1)/BLOCK_DIM);
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	ConvertKEY_Kernel<<<grid, block>>>((float4*) out->_cuData, width, height);
}


void __global__ DisplayKeyPoint_Kernel(float4 * d_result, int num)
{
	int idx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	if(idx >= num) return;
	float4 v = tex1Dfetch(texDataF4, idx);
	d_result[idx] = make_float4(v.x, v.y, 0, 1.0f);
}

void ProgramCU::DisplayKeyPoint(CuTexImage* ftex, CuTexImage* out)
{
	int num = ftex->GetImgWidth();
	int block_width = 64;
	dim3 grid((num + block_width -1) /block_width);
	dim3 block(block_width);
	ftex->BindTexture(texDataF4);
	DisplayKeyPoint_Kernel<<<grid, block>>>((float4*) out->_cuData, num);
	ProgramCU::CheckErrorCUDA("DisplayKeyPoint");
}

void __global__ DisplayKeyBox_Kernel(float4* d_result, int num)
{
	int idx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	if(idx >= num) return;
	int  kidx = idx / 10, vidx = idx - IMUL(kidx , 10);
	float4 v = tex1Dfetch(texDataF4, kidx);
	float sz = fabs(v.z * 3.0f);
	///////////////////////
	float s, c;	__sincosf(v.w, &s, &c);
	///////////////////////
	float dx = vidx == 0? 0 : ((vidx <= 4 || vidx >= 9)? sz : -sz);
	float dy = vidx <= 1? 0 : ((vidx <= 2 || vidx >= 7)? -sz : sz);
	float4 pos;
	pos.x = v.x + c * dx - s * dy;
	pos.y = v.y + c * dy + s * dx;
	pos.z = 0;	pos.w = 1.0f;
	d_result[idx]  = pos;
}

void ProgramCU::DisplayKeyBox(CuTexImage* ftex, CuTexImage* out)
{
	int len = ftex->GetImgWidth();
	int block_width = 32;
	dim3 grid((len * 10 + block_width -1) / block_width);
	dim3 block(block_width);
	ftex->BindTexture(texDataF4);
	DisplayKeyBox_Kernel<<<grid, block>>>((float4*) out->_cuData, len * 10);
}
///////////////////////////////////////////////////////////////////
inline void CuTexImage:: BindTexture(textureReference& texRef)
{
	cudaBindTexture(NULL, &texRef, _cuData, &texRef.channelDesc, _numBytes);
}

inline void CuTexImage::BindTexture2D(textureReference& texRef)
{
#if defined(SIFTGPU_ENABLE_LINEAR_TEX2D) 
	cudaBindTexture2D(0, &texRef, _cuData, &texRef.channelDesc, _imgWidth, _imgHeight, _imgWidth* _numChannel* sizeof(float));
#else
	cudaChannelFormatDesc desc;
	cudaGetChannelDesc(&desc, _cuData2D);
	cudaBindTextureToArray(&texRef, _cuData2D, &desc);
#endif
}

int ProgramCU::CheckCudaDevice(int device)
{
	int count = 0, device_used; 
	if(cudaGetDeviceCount(&count) != cudaSuccess  || count <= 0)
	{
		ProgramCU::CheckErrorCUDA("CheckCudaDevice");
		return 0;
	}else if(count == 1)
	{
		cudaDeviceProp deviceProp;
		if ( cudaGetDeviceProperties(&deviceProp, 0) != cudaSuccess  ||
			(deviceProp.major == 9999 && deviceProp.minor == 9999))
		{
			fprintf(stderr, "CheckCudaDevice: no device supporting CUDA.\n");
			return 0;
		}else
		{
			GlobalUtil::_MemCapGPU = deviceProp.totalGlobalMem / 1024;
			GlobalUtil::_texMaxDimGL = 132768;
			if(GlobalUtil::_verbose) 
				fprintf(stdout, "NOTE: changing maximum texture dimension to %d\n", GlobalUtil::_texMaxDimGL);

		}
	}
	if(device >0 && device < count)  
	{
		cudaSetDevice(device);
		CheckErrorCUDA("cudaSetDevice\n"); 
	}
	cudaGetDevice(&device_used);
	if(device != device_used) 
		fprintf(stderr,  "\nERROR:   Cannot set device to %d\n"
		"\nWARNING: Use # %d device instead (out of %d)\n", device, device_used, count);
	return 1;
}

////////////////////////////////////////////////////////////////////////////////////////
// siftmatch funtions
//////////////////////////////////////////////////////////////////////////////////////////

#define MULT_TBLOCK_DIMX 128
#define MULT_TBLOCK_DIMY 1
#define MULT_BLOCK_DIMX (MULT_TBLOCK_DIMX)
#define MULT_BLOCK_DIMY (8 * MULT_TBLOCK_DIMY)


texture<uint4, 1, cudaReadModeElementType> texDes1;
texture<uint4, 1, cudaReadModeElementType> texDes2;


//dim grid(num2/128,num1/8)
//dim block(128,1)
void __global__ MultiplyDescriptor_Kernel(int* d_result, int num1, int num2, int3* d_temp)
{
	//MULT_BLOCK_DIMY : 8         MULT_BLOCK_DIMX : 128
	int idx01 = (blockIdx.y  * MULT_BLOCK_DIMY),  //0~num1(0,8,16,24,32...)
		idx02 = (blockIdx.x  * MULT_BLOCK_DIMX);  //0~num2
	int idx1 = idx01 + threadIdx.y, //idx1 = idx01
		idx2 = idx02 + threadIdx.x;//col线程的列索引
	__shared__ int data1[17 * 2 * MULT_BLOCK_DIMY];  //每个线程块共享内存272，共享内存是一维的！ 线程块大小：（128,1）
	int read_idx1 = idx01 * 8 +  threadIdx.x,//用到了共享内存
		read_idx2 = idx2 * 8;//没用到共享内存
	int col4 = threadIdx.x & 0x3, //得到0,1,2,3
		row4 = threadIdx.x >> 2; //threadIdx.x/4
	int cache_idx1 = IMUL(row4, 17) + (col4 << 2);//row4*17+col4*4

	///////////////////////////////////////////////////////////////
	//Load feature descriptors
	///////////////////////////////////////////////////////////////
#if MULT_BLOCK_DIMY == 16
	uint4 v = tex1Dfetch(texDes1, read_idx1);
	data1[cache_idx1]   = v.x;	data1[cache_idx1+1] = v.y;
	data1[cache_idx1+2] = v.z;	data1[cache_idx1+3] = v.w;
#elif MULT_BLOCK_DIMY == 8
	if(threadIdx.x < 64) //threadIdx.x = 64时！！！cache_idx1为272！！！
	{
		uint4 v = tex1Dfetch(texDes1, read_idx1); //num1的索引
		data1[cache_idx1]   = v.x;		data1[cache_idx1+1] = v.y;
		data1[cache_idx1+2] = v.z;		data1[cache_idx1+3] = v.w;
	}
#else
#error
#endif
	__syncthreads();

	///
	if(idx2 >= num2) return;
	///////////////////////////////////////////////////////////////////////////
	//compare descriptors

	int results[MULT_BLOCK_DIMY]; //8个result
#pragma unroll
	for(int i = 0; i < MULT_BLOCK_DIMY; ++i) results[i] = 0;

#pragma unroll
	for(int i = 0; i < 8; ++i)  //8
	{
		uint4 v = tex1Dfetch(texDes2, read_idx2 + i);
		unsigned char* p2 = (unsigned char*)(&v);  //取出int包含了16个char
#pragma unroll
		for(int k = 0; k < MULT_BLOCK_DIMY; ++k)  //8
		{
			unsigned char* p1 = (unsigned char*) (data1 + k * 34 + i *  4 + (i/4)); //i/4是余数，因为每隔4个就加5
			results[k] += 	 ( IMUL(p1[0], p2[0])	+ IMUL(p1[1], p2[1])  
				+ IMUL(p1[2], p2[2])  	+ IMUL(p1[3], p2[3])  
				+ IMUL(p1[4], p2[4])  	+ IMUL(p1[5], p2[5])  
				+ IMUL(p1[6], p2[6])  	+ IMUL(p1[7], p2[7])  
				+ IMUL(p1[8], p2[8])  	+ IMUL(p1[9], p2[9])  
				+ IMUL(p1[10], p2[10])	+ IMUL(p1[11], p2[11])
				+ IMUL(p1[12], p2[12])	+ IMUL(p1[13], p2[13])
				+ IMUL(p1[14], p2[14])	+ IMUL(p1[15], p2[15]));

				//		results[k] += 	 ( IMUL(p1[0]-p2[0], p1[0]-p2[0])	+ IMUL(p1[1]-p2[1], p1[1]-p2[1])  
				//+ IMUL(p1[2]-p2[2], p1[2]-p2[2])  	+ IMUL(p1[3]-p2[3],p1[3]-p2[3] )  
				//+ IMUL(p1[4]-p2[4], p1[4]-p2[4])  	+ IMUL(p1[5]-p2[5], p1[5]-p2[5])  
				//+ IMUL(p1[6]-p2[6], p1[6]-p2[6])  	+ IMUL(p1[7]-p2[7], p1[7]-p2[7])  
				//+ IMUL(p1[8]-p2[8], p1[8]-p2[8])  	+ IMUL(p1[9]-p2[9], p1[9]-p2[9])  
				//+ IMUL(p1[10]-p2[10],p1[10]-p2[10])	+ IMUL(p1[11]-p2[11], p1[11]-p2[11])
				//+ IMUL(p1[12]-p2[12], p1[12]-p2[12])	+ IMUL(p1[13] -p2[13],p1[13] -p2[13])
				//+ IMUL(p1[14]-p2[14],p1[14]-p2[14] )	+ IMUL(p1[15]-p2[15], p1[15]-p2[15]));
		}
	}

	int dst_idx = IMUL(idx1, num2)  + idx2;  //(8*threadIdx.y*num2+idx2)
	if(d_temp)
	{
		int3 cmp_result = make_int3(0, -1, 0);  //8个result里面(最大距离，位置(num1上的)，次大值)

#pragma unroll
		for(int i = 0; i < MULT_BLOCK_DIMY; ++i)  //8
		{
			if(idx1 + i < num1)//i:0~8+idx1
			{
				cmp_result = results[i] > cmp_result.x? 	make_int3(results[i], idx1 + i, cmp_result.x) : 	make_int3(cmp_result.x, cmp_result.y, max(cmp_result.z, results[i]));
				//(i, num2)由于是共性内存，所以用了八个
				d_result[dst_idx + IMUL(i, num2)] = results[i];
			}
		}
		d_temp[ IMUL(blockIdx.y, num2) + idx2] = cmp_result; 
	}
	else
	{
#pragma unroll
		for(int i = 0; i < MULT_BLOCK_DIMY; ++i)//8
		{
			if(idx1 + i < num1) 
				d_result[dst_idx + IMUL(i, num2)] = results[i];
		}
	}

}


void ProgramCU::MultiplyDescriptor(CuTexImage* des1, CuTexImage* des2, CuTexImage* texDot, CuTexImage* texCRT)
{
	int num1 = des1->GetImgWidth() / 8;	//1067  num*8  4通道 float4
	int num2 = des2->GetImgWidth() / 8;   //728
	dim3 grid(	(num2 + MULT_BLOCK_DIMX - 1)/ MULT_BLOCK_DIMX,
		(num1 + MULT_BLOCK_DIMY - 1)/MULT_BLOCK_DIMY);// (num2/128,num1/8)
	dim3 block(MULT_TBLOCK_DIMX, MULT_TBLOCK_DIMY);//(128,1)
	//使用的线程的大小为(num1/8)*num2

	texDot->InitTexture( num2,num1);// (num2*num1)*4  float4

	if(texCRT)        //(num2 * num1/8 ) *4 float4
		texCRT->InitTexture(num2, (num1 + MULT_BLOCK_DIMY - 1)/MULT_BLOCK_DIMY, 32);
	des1->BindTexture(texDes1);//
	des2->BindTexture(texDes2);//


	//输入：CRT或NULL   texDes1和texDes2
	//输出：texDot

		GlobalUtil::StartTimer("竖直");
	MultiplyDescriptor_Kernel<<<grid, block>>>((int*)texDot->_cuData, num1, num2, 
		(texCRT? (int3*)texCRT->_cuData : NULL));
				cudaThreadSynchronize();			
				GlobalUtil::StopTimer();

		float _timing1 = GlobalUtil::GetElapsedTime();

				cudaThreadSynchronize();			


	ProgramCU::CheckErrorCUDA("MultiplyDescriptor");
}

texture<float, 1, cudaReadModeElementType> texLoc1;
texture<float2, 1, cudaReadModeElementType> texLoc2;
struct Matrix33{float mat[3][3];};


 //(num2/128，num1/8)
 //(128,8)
void __global__ MultiplyDescriptorG_Kernel(int* d_result, int num1, int num2, int3* d_temp,
	Matrix33 H, float hdistmax, Matrix33 F, float fdistmax)
{
	int idx01 = (blockIdx.y  * MULT_BLOCK_DIMY);	//8  （0,8,16,24,32...）
	int idx02 = (blockIdx.x  * MULT_BLOCK_DIMX);//128 (0,128,256...)

	int idx1 = idx01 + threadIdx.y;//行索引
	int idx2 = idx02 + threadIdx.x;//列索引
	__shared__ int data1[17 * 2 * MULT_BLOCK_DIMY];//272
	__shared__ float loc1[MULT_BLOCK_DIMY * 2];  //16
	int read_idx1 = idx01 * 8 +  threadIdx.x ; //8个代表一个描述子，
	int read_idx2 = idx2 * 8;  //8个代表一个描述子，
	int col4 = threadIdx.x & 0x3, row4 = threadIdx.x >> 2;
	int cache_idx1 = IMUL(row4, 17) + (col4 << 2);
#if MULT_BLOCK_DIMY == 16   //不执行
	uint4 v = tex1Dfetch(texDes1, read_idx1);
	data1[cache_idx1]   = v.x;
	data1[cache_idx1+1] = v.y;
	data1[cache_idx1+2] = v.z;
	data1[cache_idx1+3] = v.w; 
#elif MULT_BLOCK_DIMY == 8
	if(threadIdx.x < 64)  //小于64刚好272
	{
		uint4 v = tex1Dfetch(texDes1, read_idx1);
		data1[cache_idx1]   = v.x;
		data1[cache_idx1+1] = v.y;
		data1[cache_idx1+2] = v.z;
		data1[cache_idx1+3] = v.w;
	}
#else
#error
#endif
	__syncthreads();
	if(threadIdx.x < MULT_BLOCK_DIMY * 2) //小于16
	{
		// （0,8,16,24,32...）->（0,16,32,48,64...）+16
		loc1[threadIdx.x] = tex1Dfetch(texLoc1, 2 * idx01 + threadIdx.x);  //一个在共享内存里
	}
	__syncthreads();
	if(idx2 >= num2) return;
	int results[MULT_BLOCK_DIMY];
	/////////////////////////////////////////////////////////////////////////////////////////////
	//geometric verification    几何验证
	/////////////////////////////////////////////////////////////////////////////////////////////
	int good_count = 0;
	float2 loc2 = tex1Dfetch(texLoc2, idx2);  //一个在纹理内存里，负责定位   
	//idx2=
	//每个特征点对应
#pragma unroll
	for(int i = 0; i < MULT_BLOCK_DIMY; ++i)//8
	{
		if(idx1 + i < num1)
		{
			float* loci = loc1 + i * 2;
			float locx = loci[0], locy = loci[1];
			//homography
			float x[3], diff[2];
			x[0] = H.mat[0][0] * locx + H.mat[0][1] * locy + H.mat[0][2];
			x[1] = H.mat[1][0] * locx + H.mat[1][1] * locy + H.mat[1][2];
			x[2] = H.mat[2][0] * locx + H.mat[2][1] * locy + H.mat[2][2];
			diff[0] = fabs(FDIV(x[0], x[2]) - loc2.x);
			diff[1] = fabs(FDIV(x[1], x[2]) - loc2.y);
			if(diff[0] < hdistmax && diff[1] < hdistmax)
			{
				//check fundamental matrix 检查基础矩阵
				float fx1[3], ftx2[3], x2fx1, se; 
				fx1[0] = F.mat[0][0] * locx + F.mat[0][1] * locy + F.mat[0][2];
				fx1[1] = F.mat[1][0] * locx + F.mat[1][1] * locy + F.mat[1][2];
				fx1[2] = F.mat[2][0] * locx + F.mat[2][1] * locy + F.mat[2][2];

				ftx2[0] = F.mat[0][0] * loc2.x + F.mat[1][0] * loc2.y + F.mat[2][0];
				ftx2[1] = F.mat[0][1] * loc2.x + F.mat[1][1] * loc2.y + F.mat[2][1];
				//ftx2[2] = F.mat[0][2] * loc2.x + F.mat[1][2] * loc2.y + F.mat[2][2];

				x2fx1 = loc2.x * fx1[0]  + loc2.y * fx1[1] + fx1[2];
				se = FDIV(x2fx1 * x2fx1, fx1[0] * fx1[0] + fx1[1] * fx1[1] + ftx2[0] * ftx2[0] + ftx2[1] * ftx2[1]);
				results[i] = se < fdistmax? 0: -262144;
			}else
			{
				results[i] = -262144;
			}
		}else
		{
			results[i] = -262144;
		}
		good_count += (results[i] >=0);
	}
	/////////////////////////////////////////////////////////////////////////////////////////////
	///compare feature descriptors anyway 无论如何都要进行特征描述子生成，跟双向匹配没有任何区别！！！
	/////////////////////////////////////////////////////////////////////////////////////////////
	if(good_count > 0)
	{
#pragma unroll
		for(int i = 0; i < 8; ++i)
		{
			uint4 v = tex1Dfetch(texDes2, read_idx2 + i);
			unsigned char* p2 = (unsigned char*)(&v);
#pragma unroll
			for(int k = 0; k < MULT_BLOCK_DIMY; ++k)
			{
				unsigned char* p1 = (unsigned char*) (data1 + k * 34 + i *  4 + (i/4));
				results[k] += 	 ( IMUL(p1[0], p2[0])	+ IMUL(p1[1], p2[1])  
					+ IMUL(p1[2], p2[2])  	+ IMUL(p1[3], p2[3])  
					+ IMUL(p1[4], p2[4])  	+ IMUL(p1[5], p2[5])  
					+ IMUL(p1[6], p2[6])  	+ IMUL(p1[7], p2[7])  
					+ IMUL(p1[8], p2[8])  	+ IMUL(p1[9], p2[9])  
					+ IMUL(p1[10], p2[10])	+ IMUL(p1[11], p2[11])
					+ IMUL(p1[12], p2[12])	+ IMUL(p1[13], p2[13])
					+ IMUL(p1[14], p2[14])	+ IMUL(p1[15], p2[15]));
			}
		}
	}
	int dst_idx = IMUL(idx1, num2)  + idx2;
	if(d_temp)
	{
		int3 cmp_result = make_int3(0, -1, 0);
#pragma unroll
		for(int i= 0; i < MULT_BLOCK_DIMY; ++i)
		{
			if(idx1 + i < num1)
			{
				cmp_result = results[i] > cmp_result.x? 
					make_int3(results[i], idx1 + i, cmp_result.x) : 
				make_int3(cmp_result.x, cmp_result.y, max(cmp_result.z, results[i]));
				d_result[dst_idx + IMUL(i, num2)] = max(results[i], 0);
			}else
			{
				break;
			}
		}
		d_temp[ IMUL(blockIdx.y, num2) + idx2] = cmp_result; 
	}else
	{
#pragma unroll
		for(int i = 0; i < MULT_BLOCK_DIMY; ++i)
		{
			if(idx1 + i < num1) d_result[dst_idx + IMUL(i, num2)] = max(results[i], 0);
			else break;
		}
	}

}



void ProgramCU::MultiplyDescriptorG(CuTexImage* des1, CuTexImage* des2,
	CuTexImage* loc1, CuTexImage* loc2, CuTexImage* texDot, CuTexImage* texCRT,
	float H[3][3], float hdistmax, float F[3][3], float fdistmax)
{
	int num1 = des1->GetImgWidth() / 8;	
	int num2 = des2->GetImgWidth() / 8;
	Matrix33 MatF, MatH;
	//copy the matrix
	memcpy(MatF.mat, F, 9 * sizeof(float));
	memcpy(MatH.mat, H, 9 * sizeof(float));
	//thread blocks
	dim3 grid(	(num2 + MULT_BLOCK_DIMX - 1)/ MULT_BLOCK_DIMX,   //(num2/128，num1/8)
		(num1 + MULT_BLOCK_DIMY - 1)/MULT_BLOCK_DIMY);
	dim3 block(MULT_TBLOCK_DIMX, MULT_TBLOCK_DIMY);  //(128,8)
	//intermediate results 中间结果：双向匹配
	texDot->InitTexture( num2,num1);
	if(texCRT) texCRT->InitTexture( num2, (num1 + MULT_BLOCK_DIMY - 1)/MULT_BLOCK_DIMY, 3);
	loc1->BindTexture(texLoc1);	
	loc2->BindTexture(texLoc2);
	des1->BindTexture(texDes1);	//影像1
	des2->BindTexture(texDes2);//影像2
	MultiplyDescriptorG_Kernel<<<grid, block>>>((int*)texDot->_cuData, num1, num2, 
		(texCRT? (int3*)texCRT->_cuData : NULL),
		MatH, hdistmax, MatF, fdistmax);
}


texture<int,  1, cudaReadModeElementType> texDOT;

#define ROWMATCH_BLOCK_WIDTH 32
#define ROWMATCH_BLOCK_HEIGHT 1

void __global__  RowMatch_Kernel(int*d_dot, int* d_result, int num2, float distmax, float ratiomax)
{
#if ROWMATCH_BLOCK_HEIGHT == 1
	__shared__ int dotmax[ROWMATCH_BLOCK_WIDTH]; //32
	__shared__ int dotnxt[ROWMATCH_BLOCK_WIDTH];  //32
	__shared__ int dotidx[ROWMATCH_BLOCK_WIDTH];   //32
	int	row = blockIdx.y;
#else  //不使用
	__shared__ int x_dotmax[ROWMATCH_BLOCK_HEIGHT][ROWMATCH_BLOCK_WIDTH];   
	__shared__ int x_dotnxt[ROWMATCH_BLOCK_HEIGHT][ROWMATCH_BLOCK_WIDTH];
	__shared__ int x_dotidx[ROWMATCH_BLOCK_HEIGHT][ROWMATCH_BLOCK_WIDTH];
	int*	dotmax = x_dotmax[threadIdx.y];
	int*	dotnxt = x_dotnxt[threadIdx.y];
	int*	dotidx = x_dotidx[threadIdx.y];
	int row = IMUL(blockIdx.y, ROWMATCH_BLOCK_HEIGHT) + threadIdx.y;
#endif

	int base_address = IMUL(row , num2);  //dot层的索引！！！
	int t_dotmax = 0, t_dotnxt = 0, t_dotidx = -1;//最大值，次大值，索引
	for(int i = 0; i < num2; i += ROWMATCH_BLOCK_WIDTH)//步长32     说明threadIdx使用了num2/32次
	{
		if(threadIdx.x + i < num2)
		{
			int v = tex1Dfetch(texDOT, base_address + threadIdx.x + i);//d_dot[base_address + threadIdx.x + i];//
			bool test = v > t_dotmax;  //最大值
			t_dotnxt = test? t_dotmax : max(t_dotnxt, v);//次大值
			t_dotidx = test? (threadIdx.x + i) : t_dotidx; //只记录32个共享内存里最大值的索引！！！
			t_dotmax = test? v: t_dotmax;
		}
		__syncthreads();//在一个线程块之行结束后，t_dotmax已经被赋值，这个时候等待其它线程块求最大值
		//所有线程块都计算完之后才同步,，然后重复利用线程块，进行其他区域的判定！
	}
	//已经求得所有区域的最值！！！
	dotmax[threadIdx.x] = t_dotmax;
	dotnxt[threadIdx.x] = t_dotnxt;
	dotidx[threadIdx.x] = t_dotidx;
	__syncthreads();

#pragma unroll
	for(int step = ROWMATCH_BLOCK_WIDTH/2; step >0; step /= 2) //归约算法，每次分为两部分
	{
		if(threadIdx.x < step)
		{
			int v1 = dotmax[threadIdx.x], v2 = dotmax[threadIdx.x + step];
			bool test =  v2 > v1;
			dotnxt[threadIdx.x] = test? max(v1, dotnxt[threadIdx.x + step]) :max(dotnxt[threadIdx.x], v2);  //次大值和v1v2中较小的相比较
			dotidx[threadIdx.x] = test? dotidx[threadIdx.x + step] : dotidx[threadIdx.x];
			dotmax[threadIdx.x] = test? v2 : v1;
		}
		__syncthreads();
	}
	if(threadIdx.x == 0)
	{
		float dist =  acos(min(dotmax[0] * 0.000003814697265625f, 1.0));   //最大值，但是acos是减函数，所以得到的较小值
		float distn = acos(min(dotnxt[0] * 0.000003814697265625f, 1.0));   //次大值，但是acos是减函数，所以得到反而是较大值
		//float ratio = dist / distn;
		d_result[row] = (dist < distmax) && (dist < distn * ratiomax) ? dotidx[0] : -1;//
	}

}


void ProgramCU::GetRowMatch(CuTexImage* texDot, CuTexImage* texMatch, float distmax, float ratiomax)
{
	int num1 = texDot->GetImgHeight();//1068
	int num2 = texDot->GetImgWidth();//731
	dim3 grid(1, num1/ROWMATCH_BLOCK_HEIGHT);//dim3 grid(1,num1)
	dim3 block(ROWMATCH_BLOCK_WIDTH, ROWMATCH_BLOCK_HEIGHT);//dim3 block(32,1)
	texDot->BindTexture(texDOT);


				GlobalUtil::StartTimer("竖直");

	RowMatch_Kernel<<<grid, block>>>((int*)texDot->_cuData,
		(int*)texMatch->_cuData, num2, distmax, ratiomax);
		cudaThreadSynchronize();
				GlobalUtil::StopTimer();
		float _timing1 = GlobalUtil::GetElapsedTime();

}

#define COLMATCH_BLOCK_WIDTH 32

//texture<int3,  1, cudaReadModeElementType> texCT;

void __global__  ColMatch_Kernel(int3*d_crt, int* d_result, int height, int num2, float distmax, float ratiomax)
{
	int col = COLMATCH_BLOCK_WIDTH * blockIdx.x + threadIdx.x; //列
	if(col >= num2) return;
	int3 result = d_crt[col];//tex1Dfetch(texCT, col);
	int read_idx = col + num2;
	for(int i = 1; i < height; ++i, read_idx += num2)
	{
		int3 temp = d_crt[read_idx];//tex1Dfetch(texCT, read_idx);
		result = result.x < temp.x?   //取较大值
			make_int3(temp.x, temp.y, max(result.x, temp.z)) :
		make_int3(result.x, result.y, max(result.z, temp.x));
	}

	float dist =  acos(min(result.x * 0.000003814697265625f, 1.0));
	float distn = acos(min(result.z * 0.000003814697265625f, 1.0));
	//float ratio = dist / distn;
	d_result[col] = (dist < distmax) && (dist < distn * ratiomax) ? result.y : -1;//

}

void ProgramCU::GetColMatch(CuTexImage* texCRT, CuTexImage* texMatch, float distmax, float ratiomax)
{
	int height = texCRT->GetImgHeight();
	int num2 = texCRT->GetImgWidth();
	//texCRT->BindTexture(texCT);
	dim3 grid((num2 + COLMATCH_BLOCK_WIDTH -1) / COLMATCH_BLOCK_WIDTH);//num2/32
	dim3 block(COLMATCH_BLOCK_WIDTH);  //32*1
	ColMatch_Kernel<<<grid, block>>>((int3*)texCRT->_cuData, (int*) texMatch->_cuData, height, num2, distmax, ratiomax);
		cudaThreadSynchronize();
}

#endif

////////////////////////////////////////////////////////////////////////////
//	File:		SiftMatchCU.cpp
//	Author:		Changchang Wu
//	Description : implementation of the SiftMatchCU class.
//				CUDA-based implementation of SiftMatch
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
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
using namespace std;
#include "time.h"

#include "GlobalUtil.h"
#include "CuTexImage.h" 
#include "SiftGPU.h"
#include "ProgramCU.h"
#include "SiftMatchCU.h"


SiftMatchCU::SiftMatchCU(int max_sift):SiftMatchGPU()
{
	_num_sift[0] = _num_sift[1] = 0;
	_id_sift[0] = _id_sift[1] = 0;
	_have_loc[0] = _have_loc[1] = 0;
	_max_sift = max_sift <=0 ? 14096 : ((max_sift + 31)/ 32 * 32) ; 
	_initialized = 0;
}

void SiftMatchCU::SetMaxSift(int max_sift)//自己设定最大匹配数据，默认4096
{
	max_sift = ((max_sift + 31)/32)*32;
	//if(max_sift > GlobalUtil::_texMaxDimGL) max_sift = GlobalUtil::_texMaxDimGL;
	_max_sift = max_sift;
}


int  SiftMatchCU::CheckCudaDevice(int device)
{
    return ProgramCU::CheckCudaDevice(device);
}

void SiftMatchCU::InitSiftMatch()
{
	if(_initialized) return;
    GlobalUtil::_GoodOpenGL = max(GlobalUtil::_GoodOpenGL, 1); 
	_initialized = 1; 
}


void SiftMatchCU::SetDescriptors(int index, int num, const unsigned char* descriptors, int id)
{	
	if(_initialized == 0) return;
	if (index > 1) index = 1;
	if (index < 0) index = 0;
	_have_loc[index] = 0;
	//the same feature is already set
	if(id !=-1 && id == _id_sift[index]) return ;
	_id_sift[index] = id;
	if(num > _max_sift) num = _max_sift;
	_num_sift[index] = num; 
	_texDes[index].InitTexture(8 * num, 1, 4);// num*32 数据量变小了！！！从float变成了char    4->1     128/4/4=8
	_texDes[index].CopyFromHost((void*)descriptors);//只是计算了分配显存的大小
}

//num:特征点个数
void SiftMatchCU::SetDescriptors(int index, int num, const float* descriptors, int id)
{	
	if(_initialized == 0) return;
	if (index > 1) index = 1;
	if (index < 0) index = 0;
	if(num > _max_sift) num = _max_sift;

	sift_buffer.resize(num * 128 /4);//重新设定大小   int的size是4，char的size是1   128/4=32
	unsigned char * pub = (unsigned char*) &sift_buffer[0];//这里截断了！
	for(int i = 0; i < 128 * num; ++i)
	{
		pub[i] = int(512 * descriptors[i] + 0.5);  //因为descriptors进行了归一化，所以需要乘以512转换为char
	}
	SetDescriptors(index, num, pub, id);
}


void SiftMatchCU::SetFeautreLocation(int index, const float* locations, int gap)
{
	if(_num_sift[index] <=0) return;
	_texLoc[index].InitTexture(_num_sift[index], 1, 2);  //float2
	if(gap == 0)
	{
		_texLoc[index].CopyFromHost(locations);
	}else
	{
		sift_buffer.resize(_num_sift[index] * 2); //两倍的特征点的个数
		float* pbuf = (float*) (&sift_buffer[0]);
		for(int i = 0; i < _num_sift[index]; ++i)
		{
			pbuf[i*2] = *locations++;
			pbuf[i*2+1]= *locations ++;
			locations += gap;
		}
		_texLoc[index].CopyFromHost(pbuf);//loc为两倍的特征点的个数
	}
	_have_loc[index] = 1;
}

int  SiftMatchCU::GetGuidedSiftMatch(int max_match, int match_buffer[][2], float H[3][3], float F[3][3],
									 float distmax, float ratiomax, float hdistmax, float fdistmax, int mbm)
{

	if(_initialized ==0) return 0;
	if(_num_sift[0] <= 0 || _num_sift[1] <=0) return 0;
	if(_have_loc[0] == 0 || _have_loc[1] == 0) return 0;
	ProgramCU::MultiplyDescriptorG(_texDes, _texDes+1, _texLoc, _texLoc + 1,
		&_texDot, (mbm? &_texCRT: NULL), H, hdistmax, F, fdistmax);
	return GetBestMatch(max_match, match_buffer, distmax, ratiomax, mbm);
}

//匹配点，匹配点位的坐标，maximum distance of sift descriptor，maximum distance ratio，mutual best match or one way
int  SiftMatchCU::GetSiftMatch(int max_match, int match_buffer[][2], float distmax, float ratiomax, int mbm)
{
	if(_initialized ==0) return 0;
	if(_num_sift[0] <= 0 || _num_sift[1] <=0) return 0;
	//_texDes(8*num*4)
	ProgramCU::MultiplyDescriptor(_texDes, _texDes + 1, &_texDot, (mbm? &_texCRT: NULL));
	return GetBestMatch(max_match, match_buffer, distmax, ratiomax, mbm);
}


int SiftMatchCU::GetBestMatch(int max_match, int match_buffer[][2], float distmax, float ratiomax, int mbm)
{
	sift_buffer.resize(_num_sift[0] + _num_sift[1]);
	int * buffer1 =  (int*) &sift_buffer[0], * buffer2 = (int*) &sift_buffer[_num_sift[0]];
	_texMatch[0].InitTexture(_num_sift[0], 1);
	ProgramCU::GetRowMatch(&_texDot, _texMatch, distmax, ratiomax);
	_texMatch[0].CopyToHost(buffer1);
	if(mbm)
	{
		_texMatch[1].InitTexture(_num_sift[1], 1);
		//输入_texCRT，输出
		ProgramCU::GetColMatch(&_texCRT, _texMatch + 1, distmax, ratiomax);
		_texMatch[1].CopyToHost(buffer2);
	}
	int nmatch = 0, j ;


	for(int i = 0; i < _num_sift[0] && nmatch < max_match; ++i)
	{
		j = int(buffer1[i]);
		if( j>= 0 && (!mbm ||int(buffer2[j]) == i)) //如果没有mbm判定，直接j>0即可，如果有mbm判定，则还需int(buffer2[j]) == i)
		//if( j>= 0) 	不加mbm匹配点反而会变多！！！说明进行了剔除！ 因为家了限定条件int(buffer2[j]) == i
		{
			match_buffer[nmatch][0] = i;//特征点索引      行坐标
			match_buffer[nmatch][1] = j;//特征值指标      列坐标
			nmatch++;
		}
	}

	return nmatch;
}

#endif


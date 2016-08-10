////////////////////////////////////////////////////////////////////////////
//	File:		PyramidCU.cpp
//	Author:		Changchang Wu
//	Description : implementation of the PyramidCU class.
//				CUDA-based implementation of SiftPyramid
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
#include <string.h>
#include <math.h>
using namespace std;

#include "GlobalUtil.h"
#include "GLTexImage.h"
#include "CuTexImage.h" 
#include "SiftGPU.h"
#include "SiftPyramid.h"
#include "ProgramCU.h"
#include "PyramidCU.h"


//#include "imdebug/imdebuggl.h"
//#pragma comment (lib, "../lib/imdebug.lib")



#define USE_TIMING()		double t, t0, tt;
#define OCTAVE_START()		if(GlobalUtil::_timingO){	t = t0 = CLOCK();	cout<<"#"<<i+_down_sample_factor<<"\t";	}
#define LEVEL_FINISH()		if(GlobalUtil::_timingL){	ProgramCU::FinishCUDA();	tt = CLOCK();cout<<(tt-t)<<"\t";	t = CLOCK();}
#define OCTAVE_FINISH()		if(GlobalUtil::_timingO)cout<<"|\t"<<(CLOCK()-t0)<<endl;


PyramidCU::PyramidCU(SiftParam& sp) : SiftPyramid(sp)
{
	_allPyramid = NULL;
	_histoPyramidTex = NULL;
	_featureTex = NULL;
	_descriptorTex = NULL;
	_orientationTex = NULL;
	_bufferPBO = 0;
    _bufferTEX = NULL;
	_inputTex = new CuTexImage();

    /////////////////////////
    InitializeContext();
}

PyramidCU::~PyramidCU()
{
	DestroyPerLevelData();
	DestroySharedData();
	DestroyPyramidData();
	if(_inputTex) delete _inputTex;
    if(_bufferPBO) glDeleteBuffers(1, &_bufferPBO);
    if(_bufferTEX) delete _bufferTEX;
}

void PyramidCU::InitializeContext()
{
    GlobalUtil::InitGLParam(1);
    GlobalUtil::_GoodOpenGL = max(GlobalUtil::_GoodOpenGL, 1); 
}

//w:宽度
//h:高度
//ds:降采样的系数  2的ds次方！ ds降采样因子
void PyramidCU::InitPyramid(int w, int h, int ds)
{
	int wp, hp, toobig = 0;  //toobig比3200大的时候为1
	if(ds == 0)
	{
		//截断宽度为4的倍数  w & 0xfffffffc;
		TruncateWidth(w);
		//降采样影子？对应数学哪里？
		_down_sample_factor = 0;  //就是ds

		if(GlobalUtil::_octave_min_default>=0)
		{
			wp = w >> _octave_min_default;//除以2的(-_octave_min_default)次方
			hp = h >> _octave_min_default;
		}else
		{
			//can't upsample by more than 8   
			//8层以上不能再进行降采样
			_octave_min_default = max(-3, _octave_min_default);
			//
			wp = w << (-_octave_min_default);  //乘以2的(-_octave_min_default)次方
			hp = h << (-_octave_min_default);  //乘以2的(-_octave_min_default)次方
		}
		_octave_min = _octave_min_default;
	}else
	{
		//must use 0 as _octave_min; 必须设为0！
		_octave_min = 0;
		_down_sample_factor = ds;    //ds降采样因子
		w >>= ds;
		h >>= ds;
		/////

		TruncateWidth(w);  //截断为4的倍数

		wp = w;
		hp = h; 

	}
	//3200在这里！  没使用
	while(wp > GlobalUtil::_texMaxDim  || hp > GlobalUtil::_texMaxDim )
	{
		_octave_min ++;
		wp >>= 1;
		hp >>= 1;
		toobig = 1;
	}
	//这两个都没经过
	while(GlobalUtil::_MemCapGPU > 0 && GlobalUtil::_FitMemoryCap &&  (wp >_pyramid_width || hp > _pyramid_height)&& 
		max(max(wp, hp), max(_pyramid_width, _pyramid_height)) >  1024 * sqrt(GlobalUtil::_MemCapGPU / 110.0))
	{
		_octave_min ++;
		wp >>= 1;
		hp >>= 1;
		toobig = 2;
	}


	//也没使用！
	if(toobig && GlobalUtil::_verbose && _octave_min > 0)
	{
		std::cout<<(toobig == 2 ? "[**SKIP OCTAVES**]:\tExceeding Memory Cap (-nomc)\n" :
					"[**SKIP OCTAVES**]:\tReaching the dimension limit(-maxd)!\n");
	}
	//ResizePyramid(wp, hp);
	//重新定义金字塔的大小，初始化的时候并没有使用
	if( wp == _pyramid_width && hp == _pyramid_height && _allocated )
	{
		FitPyramid(wp, hp);
	}
	//这里才是入口好吗！
	//正常情况下，小于3200
	else if(GlobalUtil::_ForceTightPyramid || _allocated ==0)
	{   
		//进入
		ResizePyramid(wp, hp);
	}
	else if( wp > _pyramid_width || hp > _pyramid_height )
	{
		ResizePyramid(max(wp, _pyramid_width), max(hp, _pyramid_height));
		if(wp < _pyramid_width || hp < _pyramid_height)  FitPyramid(wp, hp);
	}
	else
	{
		//try use the pyramid allocated for large image on small input images
		FitPyramid(wp, hp);
	}
}
//进入
void PyramidCU::ResizePyramid(int w, int h)
{
	unsigned int totalkb = 0; //总的kb数，kb再换算成mb
	int _octave_num_new, input_sz, i, j;  //octave新值？ 输入的sz？ i? j?

	//已经被分配过了
	if(_pyramid_width == w && _pyramid_height == h && _allocated) return;
	//如果大于3200
	if(w > GlobalUtil::_texMaxDim || h > GlobalUtil::_texMaxDim) return ;
	//
	if(GlobalUtil::_verbose && GlobalUtil::_timingS) std::cout<<"[Allocate Pyramid]:\t" <<w<<"x"<<h<<endl;
	//first octave does not change
	_pyramid_octave_first = 0; //octave第一层不变

	
	//compute # of octaves 计算octaves

	input_sz = min(w,h) ;  //宽度和高度的较小值
	_pyramid_width =  w;
	_pyramid_height =  h;



	//reset to preset parameters
	//重置预设变量!
	//这里我们定义为-1刚开始输入的时候定义的字符串
	_octave_num_new  = GlobalUtil::_octave_num_default;

	if(_octave_num_new < 1) 
	{
		_octave_num_new = (int) floor (log ( double(input_sz))/log(2.0)) -3 ;  //这里计算要有几层！结果得出有6层
		if(_octave_num_new<1 ) _octave_num_new = 1;
	}
	//如果已经重新分配了
	if(_pyramid_octave_num != _octave_num_new)
	{
		//destroy the original pyramid if the # of octave changes
		//如果有金字塔就毁掉
		if(_octave_num >0) 
		{
			DestroyPerLevelData();
			DestroyPyramidData();
		}
		//赋值给金字塔
		_pyramid_octave_num = _octave_num_new;
	}
	//确定了octave的层数
	_octave_num = _pyramid_octave_num;

	int noct = _octave_num;   //6
	int nlev = param._level_num;//6 ？这个是怎么确定的呢？
	//gus为所有的
	//	//initialize the pyramid       金字塔数据   大小为：noct* nlev * DATA_NUM=180
	if(_allPyramid==NULL)	_allPyramid = new CuTexImage[ noct* nlev * DATA_NUM]; //原始影像
	//只是非配位置，并没有大小！！！
	CuTexImage * gus =  GetBaseLevel(_octave_min, DATA_GAUSSIAN);//第一层    1
	CuTexImage * dog =  GetBaseLevel(_octave_min, DATA_DOG);//第二层             2
	CuTexImage * got =  GetBaseLevel(_octave_min, DATA_GRAD);//第四层            4
	CuTexImage * key =  GetBaseLevel(_octave_min, DATA_KEYPOINT);//第三层     8

	////////////there could be "out of memory" happening during the allocation

	for(i = 0; i< noct; i++)  //有几层塔
	{
		int wa = ((w + 3) / 4) * 4;   //规化为4的倍数

		totalkb += ((nlev *8 -19)* (wa * h) * 4 / 1024);//总字节数   (6+5+3*2+3*4)*(wa*h)*4/1024
		for( j = 0; j< nlev; j++, gus++, dog++, got++, key++)//每层塔里有几层
		{
			//0,1,2,3,4,5
			gus->InitTexture(wa, h); //nlev   有6层
			if(j==0)continue;
			dog->InitTexture(wa, h);  //nlev -1   有5层
			if(	j >= 1 && j < 1 + param._dog_level_num)//j=1,2,3
			{
				got->InitTexture(wa, h, 2); //2 * nlev - 6=2*（nlev - 3） //got有三层
				got->InitTexture2D();
			}
			if(j > 1 && j < nlev -1)//j=2,3,4	
				key->InitTexture(wa, h, 4); // nlev -3 ; 4 * nlev - 12=4*（nlev - 4） //key有三层
		}
		w>>=1; //w/2/2/2/2
		h>>=1; //h/2/2/2/2
	}

	totalkb += ResizeFeatureStorage();

	if(ProgramCU::CheckErrorCUDA("ResizePyramid")) SetFailStatus(); 

	_allocated = 1;  //已经分配了

	if(GlobalUtil::_verbose && GlobalUtil::_timingS) std::cout<<"[Allocate Pyramid]:\t" <<(totalkb/1024)<<"MB\n";

}

void PyramidCU::FitPyramid(int w, int h)
{
	_pyramid_octave_first = 0;
	//
	_octave_num  = GlobalUtil::_octave_num_default;

	int _octave_num_max = max(1, (int) floor (log ( double(min(w, h)))/log(2.0))  -3 );

	if(_octave_num < 1 || _octave_num > _octave_num_max) 
	{
		_octave_num = _octave_num_max;
	}


	int pw = _pyramid_width>>1, ph = _pyramid_height>>1;
	while(_pyramid_octave_first + _octave_num < _pyramid_octave_num &&  
		pw >= w && ph >= h)
	{
		_pyramid_octave_first++;
		pw >>= 1;
		ph >>= 1;
	}

	//////////////////
	int nlev = param._level_num;
	CuTexImage * gus =  GetBaseLevel(_octave_min, DATA_GAUSSIAN);
	CuTexImage * dog =  GetBaseLevel(_octave_min, DATA_DOG);
	CuTexImage * got =  GetBaseLevel(_octave_min, DATA_GRAD);
	CuTexImage * key =  GetBaseLevel(_octave_min, DATA_KEYPOINT);
	for(int i = 0; i< _octave_num; i++)
	{
		int wa = ((w + 3) / 4) * 4;

		for(int j = 0; j< nlev; j++, gus++, dog++, got++, key++)
		{
			gus->InitTexture(wa, h); //nlev
			if(j==0)continue;
			dog->InitTexture(wa, h);  //nlev -1
			if(	j >= 1 && j < 1 + param._dog_level_num)
			{
				got->InitTexture(wa, h, 2); //2 * nlev - 6
				got->InitTexture2D();
			}
			if(j > 1 && j < nlev -1)	key->InitTexture(wa, h, 4); // nlev -3 ; 4 * nlev - 12
		}
		w>>=1;
		h>>=1;
	}
}

int PyramidCU::CheckCudaDevice(int device)
{
    return ProgramCU::CheckCudaDevice(device);
}

void PyramidCU::SetLevelFeatureNum(int idx, int fcount)
{
	_featureTex[idx].InitTexture(fcount, 1, 4);  //分配显存
	_levelFeatureNum[idx] = fcount;          //为当前索引的特征个数
}

int PyramidCU::ResizeFeatureStorage()
{
	int totalkb = 0;//总kb数
	if(_levelFeatureNum==NULL)	
		_levelFeatureNum = new int[_octave_num * param._dog_level_num]; //6*3=18
	std::fill(_levelFeatureNum, _levelFeatureNum+_octave_num * param._dog_level_num, 0); 

	int wmax = GetBaseLevel(_octave_min)->GetImgWidth(); //底层金字塔800
	int hmax = GetBaseLevel(_octave_min)->GetImgHeight(); //底层金字塔600
	int whmax = max(wmax, hmax);//800
	int w,  i;

	//ceil(10.5)=11
	int num = (int)ceil(log(double(whmax))/log(4.0));   //log函数,意思是4的多少次方！！！
	//num=5

	if( _hpLevelNum != num)
	{
		_hpLevelNum = num;
		if(_histoPyramidTex ) delete [] _histoPyramidTex;
		_histoPyramidTex = new CuTexImage[_hpLevelNum];
	}

	for(i = 0, w = 1; i < _hpLevelNum; i++)
	{
		_histoPyramidTex[i].InitTexture(w, whmax, 4); //把金字塔绑定到纹理内存！
		w<<=2;
	}

	// (4 ^ (_hpLevelNum) -1 / 3) pixels
	totalkb += (((1 << (2 * _hpLevelNum)) -1) / 3 * 16 / 1024);  //等比数列5层直方图的总和  1+4+16+64+256=341=(4^5-1)/3
	//initialize the feature texture
	int idx = 0, n = _octave_num * param._dog_level_num;
	if(_featureTex==NULL)
		_featureTex = new CuTexImage[n]; //特征：6*3个
	if(GlobalUtil::_MaxOrientation >1 && GlobalUtil::_OrientationPack2==0 && _orientationTex== NULL)
		_orientationTex = new CuTexImage[n];//方向：


	for(i = 0; i < _octave_num; i++)
	{
		CuTexImage * tex = GetBaseLevel(i+_octave_min);  //_MaxFeaturePercent 0.005最大的特征百分比
		int fmax = int(tex->GetImgWidth() * tex->GetImgHeight()*GlobalUtil::_MaxFeaturePercent);
		//是否超过4096，由于显存的限制，不能超过4096！
		//if(fmax > GlobalUtil::_MaxLevelFeatureNum) fmax = GlobalUtil::_MaxLevelFeatureNum;
		//else if(fmax < 32) fmax = 32;	//give it at least a space of 32 feature 至少给32个特征值
		//._dog_level_num
		for(int j = 0; j < param._dog_level_num; j++, idx++)
		{
			//特征纹理
			_featureTex[idx].InitTexture(fmax, 1, 4); //绑定纹理内存！  注意特征的宽度为图像的长乘以宽诚意特征百分比
			totalkb += fmax * 16 /1024;  //为甚麽乘以16呀！！1？？？？
			//
			if(GlobalUtil::_MaxOrientation>1 && GlobalUtil::_OrientationPack2 == 0)
			{
				//方向纹理！
				_orientationTex[idx].InitTexture(fmax, 1, 4);//方向纹理的宽度为图像的长乘以宽诚意特征百分比
				totalkb += fmax * 16 /1024;  //为甚麽乘以16呀！！1？？？？
				totalkb += fmax * 16 /1024; //为甚麽乘以16呀！！1？？？？
			}
		}
	}


	//this just need be initialized once  这仅需实例化一次
	if(_descriptorTex==NULL)
	{
		//initialize feature texture pyramid 实例化特征纹理金字塔
		int fmax = _featureTex->GetImgWidth();
		_descriptorTex = new CuTexImage;
		totalkb += ( fmax /2);
		_descriptorTex->InitTexture(fmax *128, 1, 1); //描述器纹理的大小为fmax*128
	}else
	{
		totalkb +=  _descriptorTex->GetDataSize()/1024;
	}
	return totalkb;
}

void PyramidCU::GetFeatureDescriptors() 
{
	//descriptors...
	float* pd =  &_descriptor_buffer[0];
	vector<float> descriptor_buffer2;

	//use another buffer if we need to re-order the descriptors
	if(_keypoint_index.size() > 0)
	{
		descriptor_buffer2.resize(_descriptor_buffer.size());
		pd = &descriptor_buffer2[0];
	}

	CuTexImage * got, * ftex= _featureTex;
	for(int i = 0, idx = 0; i < _octave_num; i++)
	{
		got = GetBaseLevel(i + _octave_min, DATA_GRAD) + 1;
		for(int j = 0; j < param._dog_level_num; j++, ftex++, idx++, got++)
		{
			if(_levelFeatureNum[idx]==0) continue;
            ProgramCU::ComputeDescriptor(ftex, got, _descriptorTex, IsUsingRectDescription());//process
			_descriptorTex->CopyToHost(pd); //readback descriptor
			pd += 128*_levelFeatureNum[idx];
		}
	}

	if(GlobalUtil::_timingS) ProgramCU::FinishCUDA();

	if(_keypoint_index.size() > 0)
	{
	    //put the descriptor back to the original order for keypoint list.
		for(int i = 0; i < _featureNum; ++i)
		{
			int index = _keypoint_index[i];
			memcpy(&_descriptor_buffer[index*128], &descriptor_buffer2[i*128], 128 * sizeof(float));
		}
	}

	if(ProgramCU::CheckErrorCUDA("PyramidCU::GetFeatureDescriptors")) SetFailStatus(); 
}

void PyramidCU::GenerateFeatureListTex() 
{

	vector<float> list;
	int idx = 0;
	const double twopi = 2.0*3.14159265358979323846;
	float sigma_half_step = powf(2.0f, 0.5f / param._dog_level_num);
	float octave_sigma = _octave_min>=0? float(1<<_octave_min): 1.0f/(1<<(-_octave_min));
	float offset = GlobalUtil::_LoweOrigin? 0 : 0.5f; 
	if(_down_sample_factor>0) octave_sigma *= float(1<<_down_sample_factor); 

	_keypoint_index.resize(0); // should already be 0
	for(int i = 0; i < _octave_num; i++, octave_sigma*= 2.0f)
	{
		for(int j = 0; j < param._dog_level_num; j++, idx++)
		{
			list.resize(0);
			float level_sigma = param.GetLevelSigma(j + param._level_min + 1) * octave_sigma;
			float sigma_min = level_sigma / sigma_half_step;
			float sigma_max = level_sigma * sigma_half_step;
			int fcount = 0 ;
			for(int k = 0; k < _featureNum; k++)
			{
				float * key = &_keypoint_buffer[k*4];
                float sigmak = key[2]; 
                //////////////////////////////////////
                if(IsUsingRectDescription()) sigmak = min(key[2], key[3]) / 12.0f; 

				if(   (sigmak >= sigma_min && sigmak < sigma_max)
					||(sigmak < sigma_min && i ==0 && j == 0)
					||(sigmak > sigma_max && i == _octave_num -1 && j == param._dog_level_num - 1))
				{
					//add this keypoint to the list
					list.push_back((key[0] - offset) / octave_sigma + 0.5f);
					list.push_back((key[1] - offset) / octave_sigma + 0.5f);
                    if(IsUsingRectDescription())
                    {
                        list.push_back(key[2] / octave_sigma);
                        list.push_back(key[3] / octave_sigma);
                    }else
                    {
					    list.push_back(key[2] / octave_sigma);
					    list.push_back((float)fmod(twopi-key[3], twopi));
                    }
					fcount ++;
					//save the index of keypoints
					_keypoint_index.push_back(k);
				}

			}

			_levelFeatureNum[idx] = fcount;
			if(fcount==0)continue;
			CuTexImage * ftex = _featureTex+idx;

			SetLevelFeatureNum(idx, fcount);
			ftex->CopyFromHost(&list[0]);
		}
	}

	if(GlobalUtil::_verbose)
	{
		std::cout<<"#Features:\t"<<_featureNum<<"\n";
	}

}

void PyramidCU::ReshapeFeatureListCPU() 
{
	int i, szmax =0, sz;  //szmax:特征个数最多的
	int n = param._dog_level_num*_octave_num;  //6*3
	for( i = 0; i < n; i++) 
	{
		sz = _levelFeatureNum[i];
		if(sz > szmax ) szmax = sz;  //最大值
	}
	float * buffer = new float[szmax*16];//总的buffer
	float * buffer1 = buffer;//buffer指针和和buffer1相同
	float * buffer2 = buffer + szmax*4;//buffer2指向第二层 _levelFeatureNum 205  并不是第二层哎，是重新计算第一层！
	//327~205~120~78~67~41~19~19~10~12~2~6~2


	_featureNum = 0;

#ifdef NO_DUPLICATE_DOWNLOAD
	const double twopi = 2.0*3.14159265358979323846;
	_keypoint_buffer.resize(0);
	float os = _octave_min>=0? float(1<<_octave_min): 1.0f/(1<<(-_octave_min));
	if(_down_sample_factor>0) os *= float(1<<_down_sample_factor); 
	float offset = GlobalUtil::_LoweOrigin? 0 : 0.5f;
#endif


	for(i = 0; i < n; i++) //n=18 层dog金字塔
	{
		if(_levelFeatureNum[i]==0)continue;

		_featureTex[i].CopyToHost(buffer1);  //拷贝到主机端,从显存中拷出来
		
		int fcount =0;
		float * src = buffer1;//
		float * des = buffer2;//
		const static double factor  = 2.0*3.14159265358979323846/65535.0;//    2PI/65535   
		for(int j = 0; j < _levelFeatureNum[i]; j++, src+=4)//指针移四位
		{
			unsigned short * orientations = (unsigned short*) (&src[3]); //src的第四个， orientations的第一个！
			if(orientations[0] != 65535)  //没有方向值！！！
			{
				des[0] = src[0];
				des[1] = src[1];
				des[2] = src[2];
				des[3] = float( factor* orientations[0]);//orientations只是一个百分比！！！没单位！！！
				fcount++;
				des += 4;   //指针移四位
				//4位的float截断为2位的short，为主方向和辅方向
				if(orientations[1] != 65535 && orientations[1] != orientations[0])//特征点方向个数不为0且与上一个特征方向不同
				{
					des[0] = src[0];
					des[1] = src[1];
					des[2] = src[2];
					des[3] = float(factor* orientations[1]);	
					fcount++;  //辅助方向！
					des += 4;
				}
			}
		}
		//texture size
		SetLevelFeatureNum(i, fcount);
		_featureTex[i].CopyFromHost(buffer2);//计算的得到的真正的值
		
		if(fcount == 0) continue;

#ifdef NO_DUPLICATE_DOWNLOAD   //没有重复下载
		float oss = os * (1 << (i / param._dog_level_num));//1
		_keypoint_buffer.resize((_featureNum + fcount) * 4);
		float* ds = &_keypoint_buffer[_featureNum * 4];
		float* fs = buffer2;
		for(int k = 0;  k < fcount; k++, ds+=4, fs+=4)
		{
			ds[0] = oss*(fs[0]-0.5f) + offset;	//x
			ds[1] = oss*(fs[1]-0.5f) + offset;	//y
			ds[2] = oss*fs[2];  //scale
			ds[3] = (float)fmod(twopi-fs[3], twopi);	//orientation, mirrored
		}
#endif
		_featureNum += fcount;
	}
	delete[] buffer;
	if(GlobalUtil::_verbose)
	{
		std::cout<<"#Features MO:\t"<<_featureNum<<endl;
	}
}

void PyramidCU::GenerateFeatureDisplayVBO() 
{
	//it is weried that this part is very slow.
	//use a big VBO to save all the SIFT box vertices
	int nvbo = _octave_num * param._dog_level_num;
	if(_featureDisplayVBO==NULL)
	{
		//initialize the vbos
		_featureDisplayVBO = new GLuint[nvbo];
		_featurePointVBO = new GLuint[nvbo];
		glGenBuffers(nvbo, _featureDisplayVBO);	
		glGenBuffers(nvbo, _featurePointVBO);
	}
	for(int i = 0; i < nvbo; i++)
	{
		if(_levelFeatureNum[i]<=0)continue;
		CuTexImage * ftex  = _featureTex + i;
		CuTexImage texPBO1( _levelFeatureNum[i]* 10, 1, 4, _featureDisplayVBO[i]);
		CuTexImage texPBO2(_levelFeatureNum[i], 1, 4, _featurePointVBO[i]);
		ProgramCU::DisplayKeyBox(ftex, &texPBO1);
		ProgramCU::DisplayKeyPoint(ftex, &texPBO2);	
	}
}

void PyramidCU::DestroySharedData() 
{
	//histogram reduction
	if(_histoPyramidTex)
	{
		delete[]	_histoPyramidTex;
		_hpLevelNum = 0;
		_histoPyramidTex = NULL;
	}
	//descriptor storage shared by all levels
	if(_descriptorTex)
	{
		delete _descriptorTex;
		_descriptorTex = NULL;
	}
	//cpu reduction buffer.
	if(_histo_buffer)
	{
		delete[] _histo_buffer;
		_histo_buffer = 0;
	}
}

void PyramidCU::DestroyPerLevelData() 
{
	//integers vector to store the feature numbers.
	if(_levelFeatureNum)
	{
		delete [] _levelFeatureNum;
		_levelFeatureNum = NULL;
	}
	//texture used to store features
	if(	_featureTex)
	{
		delete [] _featureTex;
		_featureTex =	NULL;
	}
	//texture used for multi-orientation 
	if(_orientationTex)
	{
		delete [] _orientationTex;
		_orientationTex = NULL;
	}
	int no = _octave_num* param._dog_level_num;

	//two sets of vbos used to display the features
	if(_featureDisplayVBO)
	{
		glDeleteBuffers(no, _featureDisplayVBO);
		delete [] _featureDisplayVBO;
		_featureDisplayVBO = NULL;
	}
	if( _featurePointVBO)
	{
		glDeleteBuffers(no, _featurePointVBO);
		delete [] _featurePointVBO;
		_featurePointVBO = NULL;
	}
}

void PyramidCU::DestroyPyramidData()
{
	if(_allPyramid)
	{
		delete [] _allPyramid;
		_allPyramid = NULL;
	}
}

void PyramidCU::DownloadKeypoints() 
{
	const double twopi = 2.0*3.14159265358979323846;
	int idx = 0;
	float * buffer = &_keypoint_buffer[0];
	vector<float> keypoint_buffer2;
	//use a different keypoint buffer when processing with an exisint features list
	//without orientation information. 
	if(_keypoint_index.size() > 0)
	{
		keypoint_buffer2.resize(_keypoint_buffer.size());
		buffer = &keypoint_buffer2[0];
	}
	float * p = buffer, *ps;
	CuTexImage * ftex = _featureTex;
	/////////////////////
	float os = _octave_min>=0? float(1<<_octave_min): 1.0f/(1<<(-_octave_min));
	if(_down_sample_factor>0) os *= float(1<<_down_sample_factor); 
	float offset = GlobalUtil::_LoweOrigin? 0 : 0.5f;
	/////////////////////
	for(int i = 0; i < _octave_num; i++, os *= 2.0f)
	{
	
		for(int j = 0; j  < param._dog_level_num; j++, idx++, ftex++)
		{

			if(_levelFeatureNum[idx]>0)
			{	
				ftex->CopyToHost(ps = p);
				for(int k = 0;  k < _levelFeatureNum[idx]; k++, ps+=4)
				{
					ps[0] = os*(ps[0]-0.5f) + offset;	//x
					ps[1] = os*(ps[1]-0.5f) + offset;	//y
					ps[2] = os*ps[2]; 
					ps[3] = (float)fmod(twopi-ps[3], twopi);	//orientation, mirrored
				}
				p+= 4* _levelFeatureNum[idx];
			}
		}
	}

	//put the feature into their original order for existing keypoint 
	if(_keypoint_index.size() > 0)
	{
		for(int i = 0; i < _featureNum; ++i)
		{
			int index = _keypoint_index[i];
			memcpy(&_keypoint_buffer[index*4], &keypoint_buffer2[i*4], 4 * sizeof(float));
		}
	}
}

void PyramidCU::GenerateFeatureListCPU()
{
	//no cpu version provided
	GenerateFeatureList();
}
//
void PyramidCU::GenerateFeatureList(int i, int j, int reduction_count, vector<int>& hbuffer)
{
	//i 0~5  j 0~2
    int fcount = 0, idx = i * param._dog_level_num  + j; //param._dog_level_num=3
	int hist_level_num = _hpLevelNum - _pyramid_octave_first /2; //_hpLevelNum=5
	int ii, k, len; 

	CuTexImage * htex, * ftex, * tex, *got;
	ftex = _featureTex + idx; //18层
	htex = _histoPyramidTex + hist_level_num -1;// hist_level_num=5  //5层
	tex = GetBaseLevel(_octave_min + i, DATA_KEYPOINT) + 2 + j;
	got = GetBaseLevel(_octave_min + i, DATA_GRAD) + 2 + j;
	//第四层200*600的那层
	ProgramCU::InitHistogram(tex, htex); //实例化直方图
	//200~50~13~4~1
	for(k = 0; k < reduction_count - 1; k++, htex--)
	{
		//本层，下一层，最高层第五层是800*600
		ProgramCU::ReduceHistogram(htex, htex -1);	//归约直方图   4归1  水平方向，四个重合成一个
	}
	//得到htex为600*1,  4通道
	//htex has the row reduction result   htex 包含是行归约结果
	len = htex->GetImgHeight() * 4;//600*4     4通道所以乘以4
	hbuffer.resize(len);//2400
	ProgramCU::FinishCUDA();
	htex->CopyToHost(&hbuffer[0]);//htex拷贝到hbuffer  GPU拷贝到CPU端！
	
    ////TO DO: track the error found here..
	for(ii = 0; ii < len; ++ii)     {if(!(hbuffer[ii]>= 0)) hbuffer[ii] = 0; }//?
	
    
    for(ii = 0; ii < len; ++ii)		
		fcount += hbuffer[ii]; //总特征个数
	SetLevelFeatureNum(idx, fcount);
	
    //build the feature list  建立特征列表
	if(fcount > 0)
	{
		_featureNum += fcount;//
		_keypoint_buffer.resize(fcount * 4); //特征点个数的四倍
		//vector<int> ikbuf(fcount*4);
		int* ibuf = (int*) (&_keypoint_buffer[0]);

		for(ii = 0; ii < len; ++ii)
		{
			int x = ii%4, y = ii / 4;
			//hbuffer[ii]不为0，也就是有特征点时ibuf才有值，
			for(int jj = 0 ; jj < hbuffer[ii]; ++jj, ibuf+=4)
			{
				//   y表示第几个特征点，x表示上一层直方图该特征点合并的位置（1或2）， ibuf[2]表示规约了几次！！！
				ibuf[0] = x; ibuf[1] = y; ibuf[2] = jj; ibuf[3] = 0;
			}
		}
		//得到_keypoint_buffer[0]刚好是327*4个，如果有两个的话循环两次，指针移动两次！
		_featureTex[idx].CopyFromHost(&_keypoint_buffer[0]);//CPU拷贝到GPU端！
	
		////////////////////////////////////////////
		//每一层特征点，对应第2层直方图也就是4*600
		//1~4~13~50~200
		ProgramCU::GenerateList(_featureTex + idx, ++htex);
		for(k = 2; k < reduction_count; k++)
		{
			ProgramCU::GenerateList(_featureTex + idx, ++htex);
		}
	}
}

//产生特征集合
void PyramidCU::GenerateFeatureList()
 {
	double t1, t2;   //time
	int ocount = 0, reduction_count; //
	//反转
    int reverse = (GlobalUtil::_TruncateMethod == 1); //截断方法

	vector<int> hbuffer;  //什么矢量
	_featureNum = 0;  //特征个数

	//for(int i = 0, idx = 0; i < _octave_num; i++)  i是0,1,2,3,4,5
    FOR_EACH_OCTAVE(i, reverse)      //6*3=18
	{
        CuTexImage* tex = GetBaseLevel(_octave_min + i, DATA_KEYPOINT) + 2; //36*2=72刚好是keypoint！
		reduction_count = FitHistogramPyramid(tex);//确定规约次数

		if(GlobalUtil::_timingO)
		{
			t1 = CLOCK(); 
			ocount = 0;
			std::cout<<"#"<<i+_octave_min + _down_sample_factor<<":\t";
		}
		//for(int j = 0; j < param._dog_level_num; j++, idx++) //j是0,1,2
        FOR_EACH_LEVEL(j, reverse)  
		{
            if(GlobalUtil::_TruncateMethod && GlobalUtil::_FeatureCountThreshold > 0 && _featureNum > GlobalUtil::_FeatureCountThreshold) continue;
			//i为octave，j为level，reduction_count为规约次数，
	        GenerateFeatureList(i, j, reduction_count, hbuffer);

			/////////////////////////////
			if(GlobalUtil::_timingO)
			{
                int idx = i * param._dog_level_num + j;
				ocount += _levelFeatureNum[idx];
				std::cout<< _levelFeatureNum[idx] <<"\t";
			}
		}
		if(GlobalUtil::_timingO)
		{	
			t2 = CLOCK(); 
			std::cout << "| \t" << int(ocount) << " :\t(" << (t2 - t1) << ")\n";
		}
	}
	/////
	CopyGradientTex();
	/////
	if(GlobalUtil::_timingS)ProgramCU::FinishCUDA();

	if(GlobalUtil::_verbose)
	{
		std::cout<<"#Features:\t"<<_featureNum<<"\n";
	}

	if(ProgramCU::CheckErrorCUDA("PyramidCU::GenerateFeatureList")) SetFailStatus();
}

GLTexImage* PyramidCU::GetLevelTexture(int octave, int level)
{
	return GetLevelTexture(octave, level, DATA_GAUSSIAN);
}

GLTexImage* PyramidCU::ConvertTexCU2GL(CuTexImage* tex, int dataName)
{
	
	GLenum format = GL_LUMINANCE;
	int convert_done = 1;
    if(_bufferPBO == 0) glGenBuffers(1, &_bufferPBO);
    if(_bufferTEX == NULL) _bufferTEX = new GLTexImage;
	switch(dataName)
	{
	case DATA_GAUSSIAN:
		{
			convert_done = tex->CopyToPBO(_bufferPBO);
			break;
		}
	case DATA_DOG:
		{
			CuTexImage texPBO(tex->GetImgWidth(), tex->GetImgHeight(), 1, _bufferPBO);
			if(texPBO._cuData == 0 || tex->_cuData == NULL) convert_done = 0;
			else ProgramCU::DisplayConvertDOG(tex, &texPBO);
			break;
		}
	case DATA_GRAD:
		{
			CuTexImage texPBO(tex->GetImgWidth(), tex->GetImgHeight(), 1, _bufferPBO);
			if(texPBO._cuData == 0 || tex->_cuData == NULL) convert_done = 0;
			else ProgramCU::DisplayConvertGRD(tex, &texPBO);
			break;
		}
	case DATA_KEYPOINT:
		{
			CuTexImage * dog = tex - param._level_num * _pyramid_octave_num;
			format = GL_RGBA;
			CuTexImage texPBO(tex->GetImgWidth(), tex->GetImgHeight(), 4, _bufferPBO);
			if(texPBO._cuData == 0 || tex->_cuData == NULL) convert_done = 0;
			else ProgramCU::DisplayConvertKEY(tex, dog, &texPBO);
			break;
		}
	default:
			convert_done = 0;
			break;
	}

	if(convert_done)
	{
		_bufferTEX->InitTexture(max(_bufferTEX->GetTexWidth(), tex->GetImgWidth()), max(_bufferTEX->GetTexHeight(), tex->GetImgHeight()));
		_bufferTEX->CopyFromPBO(_bufferPBO, tex->GetImgWidth(), tex->GetImgHeight(), format);
	}else
	{
		_bufferTEX->SetImageSize(0, 0);
	}

	return _bufferTEX;
}

GLTexImage* PyramidCU::GetLevelTexture(int octave, int level, int dataName) 
{
	CuTexImage* tex = GetBaseLevel(octave, dataName) + (level - param._level_min);
	//CuTexImage* gus = GetBaseLevel(octave, DATA_GAUSSIAN) + (level - param._level_min); 
	return ConvertTexCU2GL(tex, dataName);
}
//转为cuda操作的类型
void PyramidCU::ConvertInputToCU(GLTexInput* input)
{
	//ws宽，hs高
	int ws = input->GetImgWidth(), hs = input->GetImgHeight();
	TruncateWidth(ws); //规则化
	//copy the input image to pixel buffer object
    if(input->_pixel_data)
    {
        _inputTex->InitTexture(ws, hs, 1);  //计算分配数据大小
        _inputTex->CopyFromHost(input->_pixel_data);  //拷贝到设备
    }else
    {
        if(_bufferPBO == 0) glGenBuffers(1, &_bufferPBO);
        if(input->_rgb_converted && input->CopyToPBO(_bufferPBO, ws, hs, GL_LUMINANCE))
        {
		    _inputTex->InitTexture(ws, hs, 1);
            _inputTex->CopyFromPBO(ws, hs, _bufferPBO); 
        }else if(input->CopyToPBO(_bufferPBO, ws, hs))
	    {
		    CuTexImage texPBO(ws, hs, 4, _bufferPBO);
		    _inputTex->InitTexture(ws, hs, 1);
		    ProgramCU::ReduceToSingleChannel(_inputTex, &texPBO, !input->_rgb_converted);
	    }else
	    {
		    std::cerr<< "Unable To Convert Intput\n";
	    }
    }
}


void PyramidCU::BuildPyramid(GLTexInput * input)
{
	//计时器
	USE_TIMING();
	//
	int i, j;
	//800*600---400*300--200*150--100*75--52*37--28*18--
	for ( i = _octave_min; i < _octave_min + _octave_num; i++)  //6次
	{
		// filter_sigma 滤波σ
		float* filter_sigma = param._sigma; //高斯滤波变量σ
		//tex就是gus
		CuTexImage *tex = GetBaseLevel(i);    //1个通道
		//buf就是key，+2的意思是第一个值不为空！
		CuTexImage *buf = GetBaseLevel(i, DATA_KEYPOINT) +2; //4个通道   DATA_KEYPOINT	= 2
		j = param._level_min + 1;

		OCTAVE_START();

		if( i == _octave_min )
		{	
			//计算显存大小并分配给_cuData
			ConvertInputToCU(input);

			if(i == 0) //第一层金字塔！！！ 
			{
				//高斯滤波函数
				//参数：
				//tex输出影像，_inputTex输入影像，buf缓冲影像！
				ProgramCU::FilterImage(tex, _inputTex, buf,  //inputTex是原始影像，只有一个！！！
                    param.GetInitialSmoothSigma(_octave_min + _down_sample_factor)); //降采样因子2的多少次方
			}
			else
			{
				if(i < 0)	
					ProgramCU::SampleImageU(tex, _inputTex, -i);			//uper 上面的
				else		
					ProgramCU::SampleImageD(tex, _inputTex, i);        //down下面的
				//
				ProgramCU::FilterImage(tex, tex, buf, 
                    param.GetInitialSmoothSigma(_octave_min + _down_sample_factor));
			}
		}
		else
		{
			ProgramCU::SampleImageD(tex, GetBaseLevel(i - 1) + param._level_ds - param._level_min); 
			if(param._sigma_skip1 > 0)
			{
				ProgramCU::FilterImage(tex, tex, buf, param._sigma_skip1);
			}
		}
		//后5层的金字塔
		LEVEL_FINISH();
		for( ; j <=  param._level_max ; j++, tex++, filter_sigma++)
		{
			// filtering
			//输出下一层影像，输入该层影像，，缓冲
			ProgramCU::FilterImage(tex + 1, tex, buf, *filter_sigma);
			LEVEL_FINISH();
		}
		OCTAVE_FINISH();
	}
	if(GlobalUtil::_timingS) ProgramCU::FinishCUDA();

	if(ProgramCU::CheckErrorCUDA("PyramidCU::BuildPyramid")) SetFailStatus();
}
//计算高斯差分DOG和3层DOG检测关键点keypoint
void PyramidCU::DetectKeypointsEX()
{


	int i, j;
	double t0, t, ts, t1, t2;

	if(GlobalUtil::_timingS && GlobalUtil::_verbose)ts = CLOCK();

	for(i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		CuTexImage * gus = GetBaseLevel(i) + 1;						 //首级金字塔网上一层
		CuTexImage * dog = GetBaseLevel(i, DATA_DOG) + 1;	 //去掉0值
		CuTexImage * got = GetBaseLevel(i, DATA_GRAD) + 1;  //去掉0值
		//compute the gradient  计算梯度
		for(j = param._level_min +1; j <=  param._level_max ; j++, gus++, dog++, got++)
		{
			//input: gus and gus -1					   输入：两层高斯金字塔
			//output: gradient, dog, orientation 输出：梯度，高斯差分，方向
			//
			ProgramCU::ComputeDOG(gus, dog, got);
		}
	}
	if(GlobalUtil::_timingS && GlobalUtil::_verbose)
	{
		ProgramCU::FinishCUDA();
		t1 = CLOCK();
	}

	for ( i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		if(GlobalUtil::_timingO)
		{
			t0 = CLOCK();
			std::cout<<"#"<<(i + _down_sample_factor)<<"\t";
		}
		CuTexImage * dog = GetBaseLevel(i, DATA_DOG) + 2;
		CuTexImage * key = GetBaseLevel(i, DATA_KEYPOINT) +2;

		//j=1,j<4    j=1,2,3
		for( j = param._level_min +2; j <  param._level_max ; j++, dog++, key++)
		{
			if(GlobalUtil::_timingL)
				t = CLOCK();
			//input, dog, dog + 1, dog -1  输入三层高斯差分金字塔
			//output, key                           输出关键点
			ProgramCU::ComputeKEY(dog, key, param._dog_threshold, param._edge_threshold);
			if(GlobalUtil::_timingL)
			{
				std::cout<<(CLOCK()-t)<<"\t";
			}
		}
		if(GlobalUtil::_timingO)
		{
			std::cout<<"|\t"<<(CLOCK()-t0)<<"\n";
		}
	}

	if(GlobalUtil::_timingS)
	{
		ProgramCU::FinishCUDA();
		if(GlobalUtil::_verbose) 
		{	
			t2 = CLOCK();
			std::cout	<<"<Gradient, DOG  >\t"<<(t1-ts)<<"\n"
						<<"<Get Keypoints  >\t"<<(t2-t1)<<"\n";
		}				
	}
}

void PyramidCU::CopyGradientTex()
{
	double ts, t1;

	if(GlobalUtil::_timingS && GlobalUtil::_verbose)ts = CLOCK();

	for(int i = 0, idx = 0; i < _octave_num; i++)
	{
		CuTexImage * got = GetBaseLevel(i + _octave_min, DATA_GRAD) +  1;
		//compute the gradient
		for(int j = 0; j <  param._dog_level_num ; j++, got++, idx++)
		{
			if(_levelFeatureNum[idx] > 0)	
				got->CopyToTexture2D();
		}
	}
	if(GlobalUtil::_timingS)
	{
		ProgramCU::FinishCUDA();
		if(GlobalUtil::_verbose)
		{
			t1 = CLOCK();
			std::cout	<<"<Copy Grad/Orientation>\t"<<(t1-ts)<<"\n";
		}
	}
}

void PyramidCU::ComputeGradient() 
{

	int i, j;
	double ts, t1;

	if(GlobalUtil::_timingS && GlobalUtil::_verbose)ts = CLOCK();

	for(i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		CuTexImage * gus = GetBaseLevel(i) +  1;
		CuTexImage * dog = GetBaseLevel(i, DATA_DOG) +  1;
		CuTexImage * got = GetBaseLevel(i, DATA_GRAD) +  1;

		//compute the gradient
		for(j = 0; j <  param._dog_level_num ; j++, gus++, dog++, got++)
		{
			ProgramCU::ComputeDOG(gus, dog, got);
		}
	}
	if(GlobalUtil::_timingS)
	{
		ProgramCU::FinishCUDA();
		if(GlobalUtil::_verbose)
		{
			t1 = CLOCK();
			std::cout	<<"<Gradient, DOG  >\t"<<(t1-ts)<<"\n";
		}
	}
}

//
int PyramidCU::FitHistogramPyramid(CuTexImage* tex)
{
	CuTexImage *htex;
	int hist_level_num = _hpLevelNum - _pyramid_octave_first / 2; 
	htex = _histoPyramidTex + hist_level_num - 1;
	int w = (tex->GetImgWidth() + 2) >> 2;  //800/4=200
	int h = tex->GetImgHeight();//600
	int count = 0; 
	for(int k = 0; k < hist_level_num; k++, htex--)
	{
		//htex->SetImageSize(w, h);	
		htex->InitTexture(w, h, 4); 
		++count;
		if(w == 1)
            break;
		w = (w + 3)>>2; 
	}
	return count;
}
//获取特征方向
void PyramidCU::GetFeatureOrientations() 
{
	//cuda操作类
	CuTexImage * ftex = _featureTex;  //18层
	int * count	 = _levelFeatureNum;
	float sigma, sigma_step = powf(2.0f, 1.0f/param._dog_level_num);

	for(int i = 0; i < _octave_num; i++)
	{
		CuTexImage* got = GetBaseLevel(i + _octave_min, DATA_GRAD) + 1;  //梯度角度
		CuTexImage* key = GetBaseLevel(i + _octave_min, DATA_KEYPOINT) + 2;//关键点

		for(int j = 0; j < param._dog_level_num; j++, ftex++, count++, got++, key++)
		{
			if(*count<=0)continue;

			//if(ftex->GetImgWidth() < *count) ftex->InitTexture(*count, 1, 4);

			sigma = param.GetLevelSigma(j+param._level_min+1);
			//ftex：特征列表，got：梯度角度，key：关键点
			ProgramCU::ComputeOrientation(ftex, got, key, sigma, sigma_step, _existing_keypoints);		
		}
	}

	if(GlobalUtil::_timingS)ProgramCU::FinishCUDA();
	if(ProgramCU::CheckErrorCUDA("PyramidCU::GetFeatureOrientations")) SetFailStatus();

}

void PyramidCU::GetSimplifiedOrientation() 
{
	//no simplified orientation
	GetFeatureOrientations();
}
//在所有金字塔影像上找到需要的位置！
CuTexImage* PyramidCU::GetBaseLevel(int octave, int dataName)

{
	//没有基层
	if(octave <_octave_min || octave > _octave_min + _octave_num) return NULL;

	int offset = (_pyramid_octave_first + octave - _octave_min) * param._level_num;
	int num = param._level_num * _pyramid_octave_num;
	if (dataName == DATA_ROT) dataName = DATA_GRAD;
	return _allPyramid + num * dataName + offset;
}

#endif


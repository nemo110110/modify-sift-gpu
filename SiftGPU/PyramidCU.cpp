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

//w:���
//h:�߶�
//ds:��������ϵ��  2��ds�η��� ds����������
void PyramidCU::InitPyramid(int w, int h, int ds)
{
	int wp, hp, toobig = 0;  //toobig��3200���ʱ��Ϊ1
	if(ds == 0)
	{
		//�ضϿ��Ϊ4�ı���  w & 0xfffffffc;
		TruncateWidth(w);
		//������Ӱ�ӣ���Ӧ��ѧ���
		_down_sample_factor = 0;  //����ds

		if(GlobalUtil::_octave_min_default>=0)
		{
			wp = w >> _octave_min_default;//����2��(-_octave_min_default)�η�
			hp = h >> _octave_min_default;
		}else
		{
			//can't upsample by more than 8   
			//8�����ϲ����ٽ��н�����
			_octave_min_default = max(-3, _octave_min_default);
			//
			wp = w << (-_octave_min_default);  //����2��(-_octave_min_default)�η�
			hp = h << (-_octave_min_default);  //����2��(-_octave_min_default)�η�
		}
		_octave_min = _octave_min_default;
	}else
	{
		//must use 0 as _octave_min; ������Ϊ0��
		_octave_min = 0;
		_down_sample_factor = ds;    //ds����������
		w >>= ds;
		h >>= ds;
		/////

		TruncateWidth(w);  //�ض�Ϊ4�ı���

		wp = w;
		hp = h; 

	}
	//3200�����  ûʹ��
	while(wp > GlobalUtil::_texMaxDim  || hp > GlobalUtil::_texMaxDim )
	{
		_octave_min ++;
		wp >>= 1;
		hp >>= 1;
		toobig = 1;
	}
	//��������û����
	while(GlobalUtil::_MemCapGPU > 0 && GlobalUtil::_FitMemoryCap &&  (wp >_pyramid_width || hp > _pyramid_height)&& 
		max(max(wp, hp), max(_pyramid_width, _pyramid_height)) >  1024 * sqrt(GlobalUtil::_MemCapGPU / 110.0))
	{
		_octave_min ++;
		wp >>= 1;
		hp >>= 1;
		toobig = 2;
	}


	//Ҳûʹ�ã�
	if(toobig && GlobalUtil::_verbose && _octave_min > 0)
	{
		std::cout<<(toobig == 2 ? "[**SKIP OCTAVES**]:\tExceeding Memory Cap (-nomc)\n" :
					"[**SKIP OCTAVES**]:\tReaching the dimension limit(-maxd)!\n");
	}
	//ResizePyramid(wp, hp);
	//���¶���������Ĵ�С����ʼ����ʱ��û��ʹ��
	if( wp == _pyramid_width && hp == _pyramid_height && _allocated )
	{
		FitPyramid(wp, hp);
	}
	//���������ں���
	//��������£�С��3200
	else if(GlobalUtil::_ForceTightPyramid || _allocated ==0)
	{   
		//����
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
//����
void PyramidCU::ResizePyramid(int w, int h)
{
	unsigned int totalkb = 0; //�ܵ�kb����kb�ٻ����mb
	int _octave_num_new, input_sz, i, j;  //octave��ֵ�� �����sz�� i? j?

	//�Ѿ����������
	if(_pyramid_width == w && _pyramid_height == h && _allocated) return;
	//�������3200
	if(w > GlobalUtil::_texMaxDim || h > GlobalUtil::_texMaxDim) return ;
	//
	if(GlobalUtil::_verbose && GlobalUtil::_timingS) std::cout<<"[Allocate Pyramid]:\t" <<w<<"x"<<h<<endl;
	//first octave does not change
	_pyramid_octave_first = 0; //octave��һ�㲻��

	
	//compute # of octaves ����octaves

	input_sz = min(w,h) ;  //��Ⱥ͸߶ȵĽ�Сֵ
	_pyramid_width =  w;
	_pyramid_height =  h;



	//reset to preset parameters
	//����Ԥ�����!
	//�������Ƕ���Ϊ-1�տ�ʼ�����ʱ������ַ���
	_octave_num_new  = GlobalUtil::_octave_num_default;

	if(_octave_num_new < 1) 
	{
		_octave_num_new = (int) floor (log ( double(input_sz))/log(2.0)) -3 ;  //�������Ҫ�м��㣡����ó���6��
		if(_octave_num_new<1 ) _octave_num_new = 1;
	}
	//����Ѿ����·�����
	if(_pyramid_octave_num != _octave_num_new)
	{
		//destroy the original pyramid if the # of octave changes
		//����н������ͻٵ�
		if(_octave_num >0) 
		{
			DestroyPerLevelData();
			DestroyPyramidData();
		}
		//��ֵ��������
		_pyramid_octave_num = _octave_num_new;
	}
	//ȷ����octave�Ĳ���
	_octave_num = _pyramid_octave_num;

	int noct = _octave_num;   //6
	int nlev = param._level_num;//6 ���������ôȷ�����أ�
	//gusΪ���е�
	//	//initialize the pyramid       ����������   ��СΪ��noct* nlev * DATA_NUM=180
	if(_allPyramid==NULL)	_allPyramid = new CuTexImage[ noct* nlev * DATA_NUM]; //ԭʼӰ��
	//ֻ�Ƿ���λ�ã���û�д�С������
	CuTexImage * gus =  GetBaseLevel(_octave_min, DATA_GAUSSIAN);//��һ��    1
	CuTexImage * dog =  GetBaseLevel(_octave_min, DATA_DOG);//�ڶ���             2
	CuTexImage * got =  GetBaseLevel(_octave_min, DATA_GRAD);//���Ĳ�            4
	CuTexImage * key =  GetBaseLevel(_octave_min, DATA_KEYPOINT);//������     8

	////////////there could be "out of memory" happening during the allocation

	for(i = 0; i< noct; i++)  //�м�����
	{
		int wa = ((w + 3) / 4) * 4;   //�滯Ϊ4�ı���

		totalkb += ((nlev *8 -19)* (wa * h) * 4 / 1024);//���ֽ���   (6+5+3*2+3*4)*(wa*h)*4/1024
		for( j = 0; j< nlev; j++, gus++, dog++, got++, key++)//ÿ�������м���
		{
			//0,1,2,3,4,5
			gus->InitTexture(wa, h); //nlev   ��6��
			if(j==0)continue;
			dog->InitTexture(wa, h);  //nlev -1   ��5��
			if(	j >= 1 && j < 1 + param._dog_level_num)//j=1,2,3
			{
				got->InitTexture(wa, h, 2); //2 * nlev - 6=2*��nlev - 3�� //got������
				got->InitTexture2D();
			}
			if(j > 1 && j < nlev -1)//j=2,3,4	
				key->InitTexture(wa, h, 4); // nlev -3 ; 4 * nlev - 12=4*��nlev - 4�� //key������
		}
		w>>=1; //w/2/2/2/2
		h>>=1; //h/2/2/2/2
	}

	totalkb += ResizeFeatureStorage();

	if(ProgramCU::CheckErrorCUDA("ResizePyramid")) SetFailStatus(); 

	_allocated = 1;  //�Ѿ�������

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
	_featureTex[idx].InitTexture(fcount, 1, 4);  //�����Դ�
	_levelFeatureNum[idx] = fcount;          //Ϊ��ǰ��������������
}

int PyramidCU::ResizeFeatureStorage()
{
	int totalkb = 0;//��kb��
	if(_levelFeatureNum==NULL)	
		_levelFeatureNum = new int[_octave_num * param._dog_level_num]; //6*3=18
	std::fill(_levelFeatureNum, _levelFeatureNum+_octave_num * param._dog_level_num, 0); 

	int wmax = GetBaseLevel(_octave_min)->GetImgWidth(); //�ײ������800
	int hmax = GetBaseLevel(_octave_min)->GetImgHeight(); //�ײ������600
	int whmax = max(wmax, hmax);//800
	int w,  i;

	//ceil(10.5)=11
	int num = (int)ceil(log(double(whmax))/log(4.0));   //log����,��˼��4�Ķ��ٴη�������
	//num=5

	if( _hpLevelNum != num)
	{
		_hpLevelNum = num;
		if(_histoPyramidTex ) delete [] _histoPyramidTex;
		_histoPyramidTex = new CuTexImage[_hpLevelNum];
	}

	for(i = 0, w = 1; i < _hpLevelNum; i++)
	{
		_histoPyramidTex[i].InitTexture(w, whmax, 4); //�ѽ������󶨵������ڴ棡
		w<<=2;
	}

	// (4 ^ (_hpLevelNum) -1 / 3) pixels
	totalkb += (((1 << (2 * _hpLevelNum)) -1) / 3 * 16 / 1024);  //�ȱ�����5��ֱ��ͼ���ܺ�  1+4+16+64+256=341=(4^5-1)/3
	//initialize the feature texture
	int idx = 0, n = _octave_num * param._dog_level_num;
	if(_featureTex==NULL)
		_featureTex = new CuTexImage[n]; //������6*3��
	if(GlobalUtil::_MaxOrientation >1 && GlobalUtil::_OrientationPack2==0 && _orientationTex== NULL)
		_orientationTex = new CuTexImage[n];//����


	for(i = 0; i < _octave_num; i++)
	{
		CuTexImage * tex = GetBaseLevel(i+_octave_min);  //_MaxFeaturePercent 0.005���������ٷֱ�
		int fmax = int(tex->GetImgWidth() * tex->GetImgHeight()*GlobalUtil::_MaxFeaturePercent);
		//�Ƿ񳬹�4096�������Դ�����ƣ����ܳ���4096��
		//if(fmax > GlobalUtil::_MaxLevelFeatureNum) fmax = GlobalUtil::_MaxLevelFeatureNum;
		//else if(fmax < 32) fmax = 32;	//give it at least a space of 32 feature ���ٸ�32������ֵ
		//._dog_level_num
		for(int j = 0; j < param._dog_level_num; j++, idx++)
		{
			//��������
			_featureTex[idx].InitTexture(fmax, 1, 4); //�������ڴ棡  ע�������Ŀ��Ϊͼ��ĳ����Կ���������ٷֱ�
			totalkb += fmax * 16 /1024;  //Ϊ�������16ѽ����1��������
			//
			if(GlobalUtil::_MaxOrientation>1 && GlobalUtil::_OrientationPack2 == 0)
			{
				//��������
				_orientationTex[idx].InitTexture(fmax, 1, 4);//��������Ŀ��Ϊͼ��ĳ����Կ���������ٷֱ�
				totalkb += fmax * 16 /1024;  //Ϊ�������16ѽ����1��������
				totalkb += fmax * 16 /1024; //Ϊ�������16ѽ����1��������
			}
		}
	}


	//this just need be initialized once  �����ʵ����һ��
	if(_descriptorTex==NULL)
	{
		//initialize feature texture pyramid ʵ�����������������
		int fmax = _featureTex->GetImgWidth();
		_descriptorTex = new CuTexImage;
		totalkb += ( fmax /2);
		_descriptorTex->InitTexture(fmax *128, 1, 1); //����������Ĵ�СΪfmax*128
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
	int i, szmax =0, sz;  //szmax:������������
	int n = param._dog_level_num*_octave_num;  //6*3
	for( i = 0; i < n; i++) 
	{
		sz = _levelFeatureNum[i];
		if(sz > szmax ) szmax = sz;  //���ֵ
	}
	float * buffer = new float[szmax*16];//�ܵ�buffer
	float * buffer1 = buffer;//bufferָ��ͺ�buffer1��ͬ
	float * buffer2 = buffer + szmax*4;//buffer2ָ��ڶ��� _levelFeatureNum 205  �����ǵڶ��㰥�������¼����һ�㣡
	//327~205~120~78~67~41~19~19~10~12~2~6~2


	_featureNum = 0;

#ifdef NO_DUPLICATE_DOWNLOAD
	const double twopi = 2.0*3.14159265358979323846;
	_keypoint_buffer.resize(0);
	float os = _octave_min>=0? float(1<<_octave_min): 1.0f/(1<<(-_octave_min));
	if(_down_sample_factor>0) os *= float(1<<_down_sample_factor); 
	float offset = GlobalUtil::_LoweOrigin? 0 : 0.5f;
#endif


	for(i = 0; i < n; i++) //n=18 ��dog������
	{
		if(_levelFeatureNum[i]==0)continue;

		_featureTex[i].CopyToHost(buffer1);  //������������,���Դ��п�����
		
		int fcount =0;
		float * src = buffer1;//
		float * des = buffer2;//
		const static double factor  = 2.0*3.14159265358979323846/65535.0;//    2PI/65535   
		for(int j = 0; j < _levelFeatureNum[i]; j++, src+=4)//ָ������λ
		{
			unsigned short * orientations = (unsigned short*) (&src[3]); //src�ĵ��ĸ��� orientations�ĵ�һ����
			if(orientations[0] != 65535)  //û�з���ֵ������
			{
				des[0] = src[0];
				des[1] = src[1];
				des[2] = src[2];
				des[3] = float( factor* orientations[0]);//orientationsֻ��һ���ٷֱȣ�����û��λ������
				fcount++;
				des += 4;   //ָ������λ
				//4λ��float�ض�Ϊ2λ��short��Ϊ������͸�����
				if(orientations[1] != 65535 && orientations[1] != orientations[0])//�����㷽�������Ϊ0������һ����������ͬ
				{
					des[0] = src[0];
					des[1] = src[1];
					des[2] = src[2];
					des[3] = float(factor* orientations[1]);	
					fcount++;  //��������
					des += 4;
				}
			}
		}
		//texture size
		SetLevelFeatureNum(i, fcount);
		_featureTex[i].CopyFromHost(buffer2);//����ĵõ���������ֵ
		
		if(fcount == 0) continue;

#ifdef NO_DUPLICATE_DOWNLOAD   //û���ظ�����
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
	ftex = _featureTex + idx; //18��
	htex = _histoPyramidTex + hist_level_num -1;// hist_level_num=5  //5��
	tex = GetBaseLevel(_octave_min + i, DATA_KEYPOINT) + 2 + j;
	got = GetBaseLevel(_octave_min + i, DATA_GRAD) + 2 + j;
	//���Ĳ�200*600���ǲ�
	ProgramCU::InitHistogram(tex, htex); //ʵ����ֱ��ͼ
	//200~50~13~4~1
	for(k = 0; k < reduction_count - 1; k++, htex--)
	{
		//���㣬��һ�㣬��߲�������800*600
		ProgramCU::ReduceHistogram(htex, htex -1);	//��Լֱ��ͼ   4��1  ˮƽ�����ĸ��غϳ�һ��
	}
	//�õ�htexΪ600*1,  4ͨ��
	//htex has the row reduction result   htex �������й�Լ���
	len = htex->GetImgHeight() * 4;//600*4     4ͨ�����Գ���4
	hbuffer.resize(len);//2400
	ProgramCU::FinishCUDA();
	htex->CopyToHost(&hbuffer[0]);//htex������hbuffer  GPU������CPU�ˣ�
	
    ////TO DO: track the error found here..
	for(ii = 0; ii < len; ++ii)     {if(!(hbuffer[ii]>= 0)) hbuffer[ii] = 0; }//?
	
    
    for(ii = 0; ii < len; ++ii)		
		fcount += hbuffer[ii]; //����������
	SetLevelFeatureNum(idx, fcount);
	
    //build the feature list  ���������б�
	if(fcount > 0)
	{
		_featureNum += fcount;//
		_keypoint_buffer.resize(fcount * 4); //������������ı�
		//vector<int> ikbuf(fcount*4);
		int* ibuf = (int*) (&_keypoint_buffer[0]);

		for(ii = 0; ii < len; ++ii)
		{
			int x = ii%4, y = ii / 4;
			//hbuffer[ii]��Ϊ0��Ҳ������������ʱibuf����ֵ��
			for(int jj = 0 ; jj < hbuffer[ii]; ++jj, ibuf+=4)
			{
				//   y��ʾ�ڼ��������㣬x��ʾ��һ��ֱ��ͼ��������ϲ���λ�ã�1��2���� ibuf[2]��ʾ��Լ�˼��Σ�����
				ibuf[0] = x; ibuf[1] = y; ibuf[2] = jj; ibuf[3] = 0;
			}
		}
		//�õ�_keypoint_buffer[0]�պ���327*4��������������Ļ�ѭ�����Σ�ָ���ƶ����Σ�
		_featureTex[idx].CopyFromHost(&_keypoint_buffer[0]);//CPU������GPU�ˣ�
	
		////////////////////////////////////////////
		//ÿһ�������㣬��Ӧ��2��ֱ��ͼҲ����4*600
		//1~4~13~50~200
		ProgramCU::GenerateList(_featureTex + idx, ++htex);
		for(k = 2; k < reduction_count; k++)
		{
			ProgramCU::GenerateList(_featureTex + idx, ++htex);
		}
	}
}

//������������
void PyramidCU::GenerateFeatureList()
 {
	double t1, t2;   //time
	int ocount = 0, reduction_count; //
	//��ת
    int reverse = (GlobalUtil::_TruncateMethod == 1); //�ضϷ���

	vector<int> hbuffer;  //ʲôʸ��
	_featureNum = 0;  //��������

	//for(int i = 0, idx = 0; i < _octave_num; i++)  i��0,1,2,3,4,5
    FOR_EACH_OCTAVE(i, reverse)      //6*3=18
	{
        CuTexImage* tex = GetBaseLevel(_octave_min + i, DATA_KEYPOINT) + 2; //36*2=72�պ���keypoint��
		reduction_count = FitHistogramPyramid(tex);//ȷ����Լ����

		if(GlobalUtil::_timingO)
		{
			t1 = CLOCK(); 
			ocount = 0;
			std::cout<<"#"<<i+_octave_min + _down_sample_factor<<":\t";
		}
		//for(int j = 0; j < param._dog_level_num; j++, idx++) //j��0,1,2
        FOR_EACH_LEVEL(j, reverse)  
		{
            if(GlobalUtil::_TruncateMethod && GlobalUtil::_FeatureCountThreshold > 0 && _featureNum > GlobalUtil::_FeatureCountThreshold) continue;
			//iΪoctave��jΪlevel��reduction_countΪ��Լ������
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
//תΪcuda����������
void PyramidCU::ConvertInputToCU(GLTexInput* input)
{
	//ws��hs��
	int ws = input->GetImgWidth(), hs = input->GetImgHeight();
	TruncateWidth(ws); //����
	//copy the input image to pixel buffer object
    if(input->_pixel_data)
    {
        _inputTex->InitTexture(ws, hs, 1);  //����������ݴ�С
        _inputTex->CopyFromHost(input->_pixel_data);  //�������豸
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
	//��ʱ��
	USE_TIMING();
	//
	int i, j;
	//800*600---400*300--200*150--100*75--52*37--28*18--
	for ( i = _octave_min; i < _octave_min + _octave_num; i++)  //6��
	{
		// filter_sigma �˲���
		float* filter_sigma = param._sigma; //��˹�˲�������
		//tex����gus
		CuTexImage *tex = GetBaseLevel(i);    //1��ͨ��
		//buf����key��+2����˼�ǵ�һ��ֵ��Ϊ�գ�
		CuTexImage *buf = GetBaseLevel(i, DATA_KEYPOINT) +2; //4��ͨ��   DATA_KEYPOINT	= 2
		j = param._level_min + 1;

		OCTAVE_START();

		if( i == _octave_min )
		{	
			//�����Դ��С�������_cuData
			ConvertInputToCU(input);

			if(i == 0) //��һ������������� 
			{
				//��˹�˲�����
				//������
				//tex���Ӱ��_inputTex����Ӱ��buf����Ӱ��
				ProgramCU::FilterImage(tex, _inputTex, buf,  //inputTex��ԭʼӰ��ֻ��һ��������
                    param.GetInitialSmoothSigma(_octave_min + _down_sample_factor)); //����������2�Ķ��ٴη�
			}
			else
			{
				if(i < 0)	
					ProgramCU::SampleImageU(tex, _inputTex, -i);			//uper �����
				else		
					ProgramCU::SampleImageD(tex, _inputTex, i);        //down�����
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
		//��5��Ľ�����
		LEVEL_FINISH();
		for( ; j <=  param._level_max ; j++, tex++, filter_sigma++)
		{
			// filtering
			//�����һ��Ӱ������ò�Ӱ�񣬣�����
			ProgramCU::FilterImage(tex + 1, tex, buf, *filter_sigma);
			LEVEL_FINISH();
		}
		OCTAVE_FINISH();
	}
	if(GlobalUtil::_timingS) ProgramCU::FinishCUDA();

	if(ProgramCU::CheckErrorCUDA("PyramidCU::BuildPyramid")) SetFailStatus();
}
//�����˹���DOG��3��DOG���ؼ���keypoint
void PyramidCU::DetectKeypointsEX()
{


	int i, j;
	double t0, t, ts, t1, t2;

	if(GlobalUtil::_timingS && GlobalUtil::_verbose)ts = CLOCK();

	for(i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		CuTexImage * gus = GetBaseLevel(i) + 1;						 //�׼�����������һ��
		CuTexImage * dog = GetBaseLevel(i, DATA_DOG) + 1;	 //ȥ��0ֵ
		CuTexImage * got = GetBaseLevel(i, DATA_GRAD) + 1;  //ȥ��0ֵ
		//compute the gradient  �����ݶ�
		for(j = param._level_min +1; j <=  param._level_max ; j++, gus++, dog++, got++)
		{
			//input: gus and gus -1					   ���룺�����˹������
			//output: gradient, dog, orientation ������ݶȣ���˹��֣�����
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
			//input, dog, dog + 1, dog -1  ���������˹��ֽ�����
			//output, key                           ����ؼ���
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
//��ȡ��������
void PyramidCU::GetFeatureOrientations() 
{
	//cuda������
	CuTexImage * ftex = _featureTex;  //18��
	int * count	 = _levelFeatureNum;
	float sigma, sigma_step = powf(2.0f, 1.0f/param._dog_level_num);

	for(int i = 0; i < _octave_num; i++)
	{
		CuTexImage* got = GetBaseLevel(i + _octave_min, DATA_GRAD) + 1;  //�ݶȽǶ�
		CuTexImage* key = GetBaseLevel(i + _octave_min, DATA_KEYPOINT) + 2;//�ؼ���

		for(int j = 0; j < param._dog_level_num; j++, ftex++, count++, got++, key++)
		{
			if(*count<=0)continue;

			//if(ftex->GetImgWidth() < *count) ftex->InitTexture(*count, 1, 4);

			sigma = param.GetLevelSigma(j+param._level_min+1);
			//ftex�������б�got���ݶȽǶȣ�key���ؼ���
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
//�����н�����Ӱ�����ҵ���Ҫ��λ�ã�
CuTexImage* PyramidCU::GetBaseLevel(int octave, int dataName)

{
	//û�л���
	if(octave <_octave_min || octave > _octave_min + _octave_num) return NULL;

	int offset = (_pyramid_octave_first + octave - _octave_min) * param._level_num;
	int num = param._level_num * _pyramid_octave_num;
	if (dataName == DATA_ROT) dataName = DATA_GRAD;
	return _allPyramid + num * dataName + offset;
}

#endif


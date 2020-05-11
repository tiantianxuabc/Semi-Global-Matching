// Copyright ?Robert Spangenberg, 2014.
// See license.txt for more details

#include "StereoCommon.h"
#include "StereoSGM.h"
#include <assert.h>

#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
template <typename T>
StereoSGM<T>::StereoSGM(int i_width, int i_height, int i_maxDisp, StereoSGMParams_t i_params)
    : m_width(i_width)
    , m_height(i_height)
    , m_maxDisp(i_maxDisp)
    , m_params(i_params)
{
    m_S = (uint16*) _mm_malloc(m_width*m_height*(i_maxDisp+1)*sizeof(uint16),16);

    m_dispLeftImgUnfiltered = (float*)_mm_malloc(m_width*m_height*sizeof(float), 16);
    m_dispRightImgUnfiltered = (float*)_mm_malloc(m_width*m_height*sizeof(float), 16);
}

template <typename T>
StereoSGM<T>::~StereoSGM()
{
    if (m_S != NULL)
        _mm_free(m_S); 

    if (m_dispLeftImgUnfiltered != NULL)
        _mm_free(m_dispLeftImgUnfiltered);
    if (m_dispRightImgUnfiltered != NULL)
        _mm_free(m_dispRightImgUnfiltered);
}



template <typename T>
uint16* StereoSGM<T>::getS()
{
    return m_S;
}

template <typename T>
int StereoSGM<T>::getHeight()
{
    return m_height;
}

template <typename T>
int StereoSGM<T>::getWidth()
{
    return m_width;
}

template <typename T>
int StereoSGM<T>::getMaxDisp()
{
    return m_maxDisp;
}

template <typename T>
void StereoSGM<T>::setParams(const StereoSGMParams_t& i_params)
{
    m_params = i_params;
}


inline void swapPointers(uint16*& p1, uint16*& p2)
{
    uint16* temp = p1;
    p1 = p2;
    p2 = temp;
}

inline sint32 adaptP2(const float32& alpha, const uint16& I_p, const uint16& I_pr, const int& gamma, const int& P2min)
{
    sint32 result;
    result = (sint32)(-alpha * abs((sint32)I_p-(sint32)I_pr)+gamma);
    if (result < P2min)
        result = P2min;
    return result;
}

template <typename T>
void StereoSGM<T>::process(uint16* dsi, T* img, float32* dispLeftImg, float32* dispRightImg)
{     
	accumulateVariableParamsSSE<8>(dsi, img, m_S);   

    // median filtering preparation
    float *dispLeftImgUnfiltered;
    float *dispRightImgUnfiltered;    
  

	dispLeftImgUnfiltered = m_dispLeftImgUnfiltered;
	dispRightImgUnfiltered = m_dispRightImgUnfiltered;
   

	matchWTA_SSE(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp, m_params.Uniqueness);
	matchRight(dispLeftImgUnfiltered, dispRightImgUnfiltered, m_width, m_height);
	//matchWTARight_SSE(dispRightImgUnfiltered, m_S,m_width, m_height, m_maxDisp, m_params.Uniqueness);		

	/* subpixel refine */       
	subPixelRefine(dispLeftImgUnfiltered, m_S, m_width, m_height, m_maxDisp);
    subPixelRefine(dispRightImgUnfiltered, m_S,m_width, m_height, m_maxDisp); 

	
	median3x3_SSE(dispLeftImgUnfiltered, dispLeftImg, m_width, m_height);
    median3x3_SSE(dispRightImgUnfiltered, dispRightImg, m_width, m_height);	

	doLRCheck(dispLeftImg, dispRightImg, m_width, m_height); 
	//doRLCheck(dispRightImg, dispLeftImg, m_width, m_height);	 	
 }



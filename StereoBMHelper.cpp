#include "StereoCommon.h"
#include "StereoBMHelper.h"
#include<fstream>
#include <assert.h>
#include <cmath>
#include <limits>
#include <smmintrin.h>
#include <emmintrin.h>
#include <nmmintrin.h>
#include <string.h>

//#define USE_AVX2

// pop count LUT for for uint16
uint16 m_popcount16LUT[UINT16_MAX+1];

void fillPopCount16LUT()
{
    // popCount LUT
    for (int i=0; i < UINT16_MAX+1; i++) 
	{
        m_popcount16LUT[i] = hamDist32(i,0);
    }
}

/*计算视差空间dsi*/
void costMeasureCensus5x5Line_xyd_SSE(uint32* intermediate1, uint32* intermediate2 
    ,const int width,const int dispCount, const uint16 invalidDispValue, uint16* dsi, const int lineStart,const int lineEnd)
{
    ALIGN16 const unsigned _LUT[] = {0x02010100, 0x03020201, 0x03020201, 0x04030302};
    const __m128i xmm7 = _mm_load_si128((__m128i*)_LUT);
    const __m128i xmm6 = _mm_set1_epi32(0x0F0F0F0F);

    for (sint32 i=lineStart;i < lineEnd;i++) {/*有效值呈阶梯状分布.[0]、[0,1]、[0,1,2]、[0,1,..,62]、[0,1,..,62,63]、...[0,1,..,63]*/

		//pBase指向leftImgCensus的第i行
        uint32* pBase = intermediate1+i*width;
		//pMatchRow指向rightImgCensus的第i行
        uint32* pMatchRow = intermediate2+i*width;


        for (uint32 j=0; j < (uint32)width; j++) 
		{
			//pBaseJ指向第i行的第j个(leftImgCensus)
            uint32* pBaseJ = pBase + j;
			//pBaseJ指向第i行的第j个的前dispCount(rightImgCensus)---采样当前值和前dispCount个值。
            uint32* pMatchRowJmD = pMatchRow + j - dispCount +1;

            int d = dispCount - 1;//(d = 63);
            
			/*当d>j时，赋初值invalidDisValue(12)*/
            for (; d >(sint32)j && d >= 0;d--) 
			{
                *getDispAddr_xyd(dsi, width, dispCount, i, j, d) = invalidDispValue;  //dsi + i*(disp*width) + j*disp + k;
                pMatchRowJmD++;
            }
			
			
            int dShift4m1 = ((d-1) >> 2)*4;
            int diff = d - dShift4m1;
            // 
            if (diff != 0)
			{
                for (; diff >= 0 && d >= 0;d--,diff--) 
				{
                    uint16 cost = (uint16)POPCOUNT32(*pBaseJ ^ *pMatchRowJmD);
                    *getDispAddr_xyd(dsi,width, dispCount, i,j,d) = cost;
                    pMatchRowJmD++;
                }
            }

            // 4 costs at once
            __m128i lPoint4 = _mm_set1_epi32(*pBaseJ);
            d -= 3;

            uint16* baseAddr = getDispAddr_xyd(dsi,width, dispCount, i,j,0);
            for (; d >= 0;d-=4) 
			{
                // flip the values
                __m128i rPoint4 = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)pMatchRowJmD), 0x1b); //mask = 00 01 10 11
                _mm_storel_pi((__m64*)(baseAddr+d), _mm_castsi128_ps(popcount32_4(_mm_xor_si128(lPoint4, rPoint4),xmm7,xmm6)));
                pMatchRowJmD+=4;
            }
            
        }
    }
} 

//intermediate1 = leftImgcensus, intermediate2 = rightImgcensus
void costMeasureCensus5x5_xyd_SSE(uint32* intermediate1, uint32* intermediate2 
    , const int height,const int width, const int dispCount, const uint16 invalidDispValue, uint16* dsi)
{
    /* 前两行为空，视差空间赋值为invalidDispValue(12)*/
    for (int i=0;i<2;i++) 
	{
        for (int j=0; j < width; j++)
		{
            for (int d=0; d <= dispCount-1;d++)
			{
                *getDispAddr_xyd(dsi,width, dispCount, i, j, d) = invalidDispValue; //为dsi赋值
            }
        }
    }

	/*计算中间行(2~height-2)的dis*/
    costMeasureCensus5x5Line_xyd_SSE(intermediate1, intermediate2, width, dispCount, invalidDispValue, dsi, 2, height-2);
 
    /*  后两行为空，视差空间赋值为invalidDispValue(12)*/
    for (int i=height-2;i<height;i++) 
	{
        for (int j=0; j < width; j++)
		{
            for (int d=0; d <= dispCount-1;d++)
			{
                *getDispAddr_xyd(dsi,width, dispCount, i,j,d) = invalidDispValue;
            }
        }
    }	
}



void matchWTA_SSE(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness)
{
    const uint32 factorUniq = (uint32)(1024*uniqueness);
    const sint32 disp = maxDisp+1;
    
    // find best by WTA
    float32* pDestDisp = dispImg;
    for (sint32 i=0;i < height; i++) 
	{
        for (sint32 j=0;j < width; j++) 
		{
            // WTA on disparity values
            
            uint16* pCost = getDispAddr_xyd(dsiAgg, width, disp, i, j, 0);
            uint16* pCostBase = pCost;
            uint32 minCost = *pCost;
            uint32 secMinCost = minCost;
            int secBestDisp = 0;

            const uint32 end = MIN(disp-1,j);
            if (end == (uint32)disp-1)
			{
                uint32 bestDisp = 0;
                for (uint32 loop =0; loop < end;loop+= 8) 
				{
                    // load costs
                    const __m128i costs = _mm_load_si128((__m128i*)pCost);
                    // get minimum for 8 values
                    const __m128i b = _mm_minpos_epu16(costs);
                    const int minValue = _mm_extract_epi16(b, 0);//min value

                   if ((uint32)minValue < minCost)
				   {
                        minCost = (uint32)minValue;
                        bestDisp = _mm_extract_epi16(b, 1) + loop;//index of min value
                   }
                    pCost += 8;
                }

                // get value of second minimum
                pCost = pCostBase;
                pCost[bestDisp]=65535;                

                __m128i secMinVector = _mm_set1_epi16(-1);
                const uint16* pCostEnd = pCost+disp;
                for (; pCost < pCostEnd; pCost += 8) 
				{
                    // load costs
                    __m128i costs = _mm_load_si128((__m128i*)pCost);
                    // get minimum for 8 values
                    secMinVector = _mm_min_epu16(secMinVector,costs);
                }
                secMinCost = _mm_extract_epi16(_mm_minpos_epu16(secMinVector),0);

                pCostBase[bestDisp]=(uint16)minCost;
                
                
                if (1024*minCost <=  secMinCost*factorUniq) 
				{
                    *pDestDisp = (float)bestDisp;
                } 
				else 
				{
                    bool check = false;
                    if (bestDisp < (uint32)maxDisp-1 && pCostBase[bestDisp+1] == secMinCost) 
					{
                        check=true;
                    } 
                    if (bestDisp > 0 && pCostBase[bestDisp-1] == secMinCost)
					{
                        check=true;
                    }
                    if (!check)
					{
                        *pDestDisp = -10;
                    } 
					else
					{
                        *pDestDisp = (float)bestDisp;						
                    }
                }
                
            } 
			else
			{
                int bestDisp = 0;
                // for start
                for (uint32 k=1; k <= end; k++) 
				{
                    pCost += 1;
                    const uint16 cost = *pCost;
                    if (cost < secMinCost) 
					{
                        if (cost < minCost) 
						{
                            secMinCost = minCost;
                            secBestDisp = bestDisp;
                            minCost = cost;
                            bestDisp = k;
                        } 
						else  
						{
                            secMinCost = cost;
                            secBestDisp = k;
                        }
                    }
                }
                // assign disparity
                if (1024*minCost <=  secMinCost*factorUniq || abs(bestDisp - secBestDisp) < 2) 
				{
                    *pDestDisp = (float)bestDisp;
                } 
				else 
				{
                    *pDestDisp = -10;
                }
            }
            pDestDisp++;
        }
    }
}

FORCEINLINE __m128 rcp_nz_ss(__m128 input) {
    __m128 mask = _mm_cmpeq_ss(_mm_set1_ps(0.0), input);
    __m128 recip = _mm_rcp_ss(input);
    return _mm_andnot_ps(mask, recip);
}



void matchWTARight_SSE(float32* dispImg, uint16* &dsiAgg, const int width, const int height, const int maxDisp, const float32 uniqueness)
{
    const uint32 factorUniq = (uint32)(1024*uniqueness); 

    const uint32 disp = maxDisp+1;
    _ASSERT(disp <= 256);
    ALIGN32 uint16 store[256+32];
    store[15] = UINT16_MAX-1;
    store[disp+16] = UINT16_MAX-1;

    // find best by WTA
    float32* pDestDisp = dispImg;
    for (uint32 i=0;i < (uint32)height; i++) 
	{
        for (uint32 j=0;j < (uint32)width;j++) 
		{
            // WTA on disparity values
            int bestDisp = 0;
            uint16* pCost = getDispAddr_xyd(dsiAgg, width, disp, i,j,0);
            sint32 minCost = *pCost;
            sint32 secMinCost = minCost;
            int secBestDisp = 0;
            const uint32 maxCurrDisp = MIN(disp-1, width-1-j);

            if (maxCurrDisp == disp-1) 
			{

                // transfer to linear storage, slightly unrolled
	            for (uint32 k = 0; k <= maxCurrDisp; k += 4) 
				{
                    store[k+16]=*pCost;
                    store[k+16+1]=pCost[disp+1];
                    store[k+16+2]=pCost[2*disp+2];
                    store[k+16+3]=pCost[3*disp+3];
                    pCost += 4*disp+4;
                }
                // search in there
                uint16* pStore = &store[16];
                const uint16* pStoreEnd = pStore+disp;
                for (; pStore < pStoreEnd; pStore += 8) 
				{
                    // load costs
                    const __m128i costs = _mm_load_si128((__m128i*)pStore);
                    // get minimum for 8 values
                    const __m128i b = _mm_minpos_epu16(costs);
                    const int minValue = _mm_extract_epi16(b,0);

                    if (minValue < minCost) 
					{
                        minCost = minValue;
                        bestDisp = _mm_extract_epi16(b,1)+(int)(pStore-&store[16]);
                    }                    
                }

                // get value of second minimum
                pStore = &store[16];
                store[16+bestDisp]=65535;

                __m128i secMinVector = _mm_set1_epi16(-1);
                for (; pStore < pStoreEnd;pStore += 8) 
				{
                    // load costs
                    __m128i costs = _mm_load_si128((__m128i*)pStore);
                    // get minimum for 8 values
                    secMinVector = _mm_min_epu16(secMinVector,costs);
                }
                secMinCost = _mm_extract_epi16(_mm_minpos_epu16(secMinVector),0);


                // assign disparity
                if (1024U*minCost <=  secMinCost*factorUniq) 
				{
                    *pDestDisp = (float)bestDisp;
                } 
				else 
				{
                    bool check = (store[16+bestDisp+1] == secMinCost);
                    check = check  | (store[16+bestDisp-1] == secMinCost);
                    if (!check) 
					{
                        *pDestDisp = -10;
                    }
					else 
					{
                        *pDestDisp = (float)bestDisp;
                    }
                }
                pDestDisp++;
            } 
            else 
			{
				// border case handling
				for (uint32 k = 1; k <= maxCurrDisp; k++)
				{
					pCost += disp + 1;
					const sint32 cost = (sint32)*pCost;
					if (cost < secMinCost)
					{
						if (cost < minCost)
						{
							secMinCost = minCost;
							secBestDisp = bestDisp;
							minCost = cost;
							bestDisp = k;
						}
						else
						{
							secMinCost = cost;
							secBestDisp = k;
						}
					}
				}
				// assign disparity
				if (1024U * minCost <= factorUniq*secMinCost || abs(bestDisp - secBestDisp) < 2)
				{
					*pDestDisp = (float)bestDisp;
				}
				else
				{
					*pDestDisp = -10;
				}
				pDestDisp++;
            }
        }
    }
 }



 void matchRight(const float32* const src, float32* const dst, const int width, const int height)
 {
	 for (int v = 0; v < height; v++)
	 {
		 for (int u = 0; u < width; u++)
		 {
 			 if (src[v*width + u] == -10)
 			 {
 				 dst[v*width + u] = -10;				 
 			 }
 			 else
 			 {
 				 if (u - src[v*width + u] >= 0)
 				 {
 					 dst[v*width + (u - (int)src[v*width + u])] = src[v*width + u];
 				 }
 				 else
 				 {
 					 dst[v*width + u] = 0;
 				 }
 			 }
			 /*if (u - src[v*width + u] >= 0)
			 {
				 dst[v*width + (u - (int)src[v*width + u])] = src[v*width + u];
			 }
			 else
			 {
				 dst[v*width + u] = -10;
			 }*/
		 }
	 }
 }


/*  do a sub pixel refinement by a parabola fit (抛物线拟合)to the winning pixel and its neighbors */
void subPixelRefine(float32* dispImg, uint16* dsiImg, const sint32 width, const sint32 height, const sint32 maxDisp)
{
    const sint32 disp_n = maxDisp+1;

    /* equiangular */
 	for (sint32 y = 0; y < height; y++)
 	{
 		uint16*  cost = getDispAddr_xyd(dsiImg, width, disp_n, y, 1, 0);
 		float* disp = (float*)dispImg+y*width;
 
 		for (sint32 x = 1; x < width-1; x++, cost += disp_n)
 		{
 			if (disp[x] > 0.0) 
 			{
 				// Get minimum
                 int d_min = (int)disp[x];
 
 				// Compute the equations of the parabolic fit
 				uint16* costDmin = cost+d_min;
 				sint32 c0 = costDmin[-1], c1 = *costDmin, c2 = costDmin[1];
 
 				__m128 denom = _mm_cvt_si2ss(_mm_setzero_ps(),c2 - c0);
 				__m128 left = _mm_cvt_si2ss(_mm_setzero_ps(),c1-c0);
 				__m128 right = _mm_cvt_si2ss(_mm_setzero_ps(),c1-c2);
 				__m128 lowerMin = _mm_min_ss(left, right);
 				__m128 result = _mm_mul_ss(denom, rcp_nz_ss(_mm_mul_ss(_mm_set_ss(2.0f),lowerMin)));
 
 				__m128 baseDisp = _mm_cvt_si2ss(_mm_setzero_ps(),d_min);
 				result = _mm_add_ss(baseDisp, result);
 				_mm_store_ss(disp+x,result); 
 			} 
 			else 
 			{
 				disp[x] = -10;
 			}
 		}
 	}  
}

void doRLCheck(float32* dispRightImg, float32* dispCheckImg, const sint32 width, const sint32 height)
{
	float* dispRow = dispRightImg;
	float* dispCheckRow = dispCheckImg;
	for (sint32 i=0;i < height;i++) 
	{
		for (sint32 j=0;j < width;j++) 
		{
			const float32 baseDisp = dispRow[j];
			if (baseDisp >= 0 && j+baseDisp <= width) 
			{
				const float matchDisp = dispCheckRow[(int)(j+baseDisp)];

				sint32 diff = (sint32)(baseDisp - matchDisp);
 				if (abs(diff) > 1.0f) 
 				{
 					dispRow[j] = 0; // occluded or false match
 				}
			} 
			else 
			{
				dispRow[j] = 0;
			}
		}
		dispRow += width;
		dispCheckRow += width;
	}    
}

// void doLRCheck(float32* dispImg, float32* dispCheckImg, const sint32 width, const sint32 height)
// {
// 	float *dispRow = dispImg;
// 	float *dispCheckRow = dispCheckImg;
// 	//float m[8], Min, secMin, tmp;
// 	for (int i = 0; i < height; i++)
// 	{
// 		for (int j = 0; j < width; j++)
// 		{
// 			const float32 baseDisp = dispRow[j];
// 			if (baseDisp >= 0 && baseDisp <= j)
// 			{
// 				const float matchDisp = dispCheckRow[(int)(j - baseDisp)];
// 				sint32 diff = (sint32)(baseDisp - matchDisp);
// 				if (abs(diff) > 1.0f)
// 				{
// 					dispRow[j] = -10; // occluded or false match
// 				}
// 			}
// 			else
// 			{
// 				dispRow[j] = -10;				
// 			}
// 		}
// 		dispRow += width;
// 		dispCheckRow += width;
// 	}
// }


 void doLRCheck(float32* dispImg, float32* dispCheckImg,const sint32 width, const sint32 height)
 {
 	float *dispRow = dispImg;
 	float *dispCheckRow = dispCheckImg;
 	float m[8], Min, secMin, tmp;
 	for(int i = 0; i < height; i++)
 	{
 		for(int j = 0; j < width; j++)
 		{
 			const float32 baseDisp = dispRow[j];
 			if (baseDisp >= 0 && baseDisp <= j) 
 			{
 				const float matchDisp = dispCheckRow[(int)(j - baseDisp)];
 				float diff = (baseDisp - matchDisp);
  				if(abs(diff) > 1.0f )
  				{
  					if(i > 0 && i < height - 1 && j > 0 && j < width - 1)
  					{							
  						secMin = Min = m[0] = dispRow[j - 1];
  						m[1] = dispRow[j + 1];
  						m[2] = dispRow[j - width];
  						m[3] = dispRow[j - width - 1];
  						m[4] = dispRow[j - width + 1];
  						m[5] = dispRow[j + width];
  						m[6] = dispRow[j + width - 1];
  						m[7] = dispRow[j + width + 1];
  						for(int tmp_i = 1; i < 8; i++)
  						{
  						 	if(m[tmp_i] < Min)
  						 	{
  						 		secMin = Min;
  						 		Min = m[tmp_i];
  						 	}
  						 	else if(m[tmp_i] < secMin)
  						 	{
  						 		secMin = m[tmp_i];
  						 	}
  						 }
  						 dispRow[j] = secMin;
  					}					
  				}
 			}
 			else
 			{
 				
 				/*dispRow[j] = -10;*/
  				if(i > 0 && i < height - 1 && j > 0 && j < width - 1)
  				{
  					tmp = m[0] = dispRow[j - 1];
  					m[1] = dispRow[j + 1];
  					m[2] = dispRow[j - width];
  					m[3] = dispRow[j - width - 1];
  					m[4] = dispRow[j - width + 1];
  					m[5] = dispRow[j + width];
  					m[6] = dispRow[j + width - 1];
  					m[7] = dispRow[j + width + 1];
  					for(int tmp_i = 0; i < 7; i++)
  					{
  						tmp = dispRow[0];
  						for(int tmp_j = 1; j < 8 - i; j++)
  						{
  							if(tmp > m[tmp_j])
  							{
  								m[tmp_j - 1] = m[tmp_j];
  								m[tmp_j] = tmp;										
  							}
  							else
  							{
  								tmp = m[tmp_j];
  							}
  						}
  					}
  					dispRow[j] = (m[3] + m[4])/2;
  				}
 			}
 		}
 		dispRow += width;
 		dispCheckRow += width;
 	}
 }

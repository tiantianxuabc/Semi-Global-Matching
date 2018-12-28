// Copyright ?Robert Spangenberg, 2014.
// See license.txt for more details

#pragma once

#include <vector>
#include <list>
#include <string.h>

class StereoSGMParams_t {
public:
    uint16 P1; // +/-1 discontinuity penalty
    uint16 InvalidDispCost;  // init value for invalid disparities (half of max value seems ok)
    uint16 NoPasses; // one or two passes
    uint16 Paths; // 8, 0-4 gives 1D path, rest undefined
    float32 Uniqueness; // uniqueness ratio
    bool MedianFilter; // apply median filter
    bool lrCheck; // apply lr-check    
    
   

    // varP2 = - alpha * abs(I(x)-I(x-r))+gamma
    float32 Alpha; // variable P2 alpha
    uint16 Gamma; // variable P2 gamma
    uint16 P2min; // varP2 cannot get lower than P2min

    /* param set out of the paper from Banz 
    - noiseless (Cones): P1 = 11, P2min = 17, gamma = 35, alpha=0.5 8bit images 
    - Cones with noise: P1=20, P2min=24, gamma = 70, alpha=0.5
    */

    StereoSGMParams_t()
        : P1(7)
        ,InvalidDispCost(12) 
        ,NoPasses(2)
        ,Paths(8)
        ,Uniqueness(0.95f)
        ,MedianFilter(true)
        ,lrCheck(true)         
        ,Alpha(0.25f)
        ,Gamma(50)
        ,P2min(17)
    {

    }
} ;

// template param is image pixel type (uint8 or uint16)
template <typename T>
class StereoSGM {
private:
    int m_width;
    int m_height;
    int m_maxDisp;
    StereoSGMParams_t m_params;
    uint16* m_S;

    float32* m_dispLeftImgUnfiltered;
    float32* m_dispRightImgUnfiltered;

    // SSE version, only maximum 8 paths supported
    template <int NPaths> void accumulateVariableParamsSSE(uint16* &dsi, T* img, uint16* &S);

public:
     // SGM
    StereoSGM(int i_width, int i_height, int i_maxDisp, StereoSGMParams_t i_params);
    ~StereoSGM();

    void process(uint16* dsi, T* img, float32* dispLeftImg, float32* dispRightImg);
    

    // accumulation cube
    uint16* getS();
    // dimensions
    int getHeight();
    int getWidth();
    int getMaxDisp();
    // change params
    void setParams(const StereoSGMParams_t& i_params);
};


#include "StereoSGM.hpp"
#include "StereoSGM_SSE.hpp"

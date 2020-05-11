// Copyright ?Robert Spangenberg, 2014.
// See license.txt for more details


#pragma once

#include "StereoCommon.h"


/* Optimized Versions */

void census5x5_16bit_SSE(uint16* source, uint32* dest, uint32 width, uint32 height);

/* median */
void median3x3_SSE(float32* source, float32* dest, uint32 width, uint32 height);


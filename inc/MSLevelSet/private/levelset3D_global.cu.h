//============================================================================
/* Copyright (c) The Technical University of Denmark
 * Author: Ojaswa Sharma
 * E-mail: os@imm.dtu.dk
 * File: levelset3D_global.cu
 * CUDA global texture declarations.
 */
//============================================================================
#ifndef _LEVELSET3D_GLOBAL_H_
#define _LEVELSET3D_GLOBAL_H_

#include <cutil_math.h>

texture<float, 3, cudaReadModeElementType> texPhi; // 3D texture for phi
texture<float, 3, cudaReadModeElementType> texVol; // 3D texture for intensity

#endif

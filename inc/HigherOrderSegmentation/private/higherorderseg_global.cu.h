//============================================================================
/* Copyright (c) The Technical University of Denmark
 * Developed at the Computational Visualization Center (CVC), The University
 * of Texas at Austin Author: Ojaswa Sharma E-mail: os@imm.dtu.dk File:
 * higherorderseg_global.cu CUDA global texture declarations.
 */
//============================================================================
#ifndef _HIGHER_ORDER_SEG_GLOBAL_H_
#define _HIGHER_ORDER_SEG_GLOBAL_H_

#include <cutil_math.h>

texture<float, 3, cudaReadModeElementType> texPhi; // 3D texture for phi
texture<float, 3, cudaReadModeElementType> texVol; // 3D texture for intensity
texture<float, 3, cudaReadModeElementType>
    texCoeff; // 3D texture for cubic coefficient computation

#endif

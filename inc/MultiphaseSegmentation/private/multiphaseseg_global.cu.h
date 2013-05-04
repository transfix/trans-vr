//============================================================================
/* Copyright (c) The Technical University of Denmark
 * Developed at the Computational Visualization Center (CVC), The University of Texas at Austin
 * Author: Ojaswa Sharma
 * E-mail: os@imm.dtu.dk
 * File: multiphaseseg_global.cu
 * CUDA global texture declarations.
 */
//============================================================================
#ifndef _MULTIPHASE_SEG_GLOBAL_H_
#define _MULTIPHASE_SEG_GLOBAL_H_

#include <cutil_math.h>

texture<float, 3, cudaReadModeElementType> texPhi; //Pointer to 3D texture reference for phi. This is a concatenated phi texture [phi1][phi2]...[phin] so that we avoid using 
//multiple texture references for different phi volumes. Please note that with CUDA 2.1, dynamic allocation of texture arrays is not supported!

texture<float, 3, cudaReadModeElementType> texVol; //3D texture for intensity
texture<float, 3, cudaReadModeElementType> texCoeff; //3D texture for cubic coefficient computation
 
texture<float4, 1, cudaReadModeElementType> texInterfaceEllipseCenters; //Init interface ellipse centers.
texture<float, 1, cudaReadModeElementType> texAverageValues;

texture<float4, 1, cudaReadModeElementType> texCoord; //3D float3 coordinate texture for general purpose use
#endif

/******************************************************************************
				Copyright   

This code is developed within the Computational Visualization Center at The 
University of Texas at Austin.

This code has been made available to you under the auspices of a Lesser General 
Public License (LGPL) (http://www.ices.utexas.edu/cvc/software/license.html) 
and terms that you have agreed to.

Upon accepting the LGPL, we request you agree to acknowledge the use of use of 
the code that results in any published work, including scientific papers, 
films, and videotapes by citing the following references:

C. Bajaj, Z. Yu, M. Auer
Volumetric Feature Extraction and Visualization of Tomographic Molecular Imaging
Journal of Structural Biology, Volume 144, Issues 1-2, October 2003, Pages 
132-143.

If you desire to use this code for a profit venture, or if you do not wish to 
accept LGPL, but desire usage of this code, please contact Chandrajit Bajaj 
(bajaj@ices.utexas.edu) at the Computational Visualization Center at The 
University of Texas at Austin for a different license.
******************************************************************************/


#ifndef _COMP_H_
#define _COMP_H_
#include <math.h> 
#include <duallib/cellQueue.h>
#include <stdio.h> 


typedef struct _vtx { 
  float x; 
  float y; 
  float z; 
} Vtx; 
 

// bishoulder computation
void GetBishoulder(float* val , Vtx& bishoulder , float isovalue) ;
void Norm2Read_Bishoulder2(Vtx normalized_bishoulder,float* bishoulder, 
			   float xyz[3], float cell_size[3]);

bool is_ambiguous(float* val, float iso_val);





#endif

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

#ifndef C2C_CODEC_H
#define C2C_CODEC_H

#include <c2c_codec/ContourGeom.h>
#include <libdjvupp/ByteStream.h>

ContourGeom* decodeC2CFile(const char *fileName, bool & color);
ContourGeom* decodeC2CBuffer(void *data, int size, unsigned char type,
														bool & color);

void encodeC2CFile(const char *inFile, const char *outFile, unsigned char type,
										float isoval);
void writeC2CFile(void *data, unsigned char *red, unsigned char *green,
								unsigned char *blue, unsigned char type, const char *outFile,
								float isoval, int dim[3], float orig[3], float span[3]);
ByteStream *encodeC2CBuffer(void *data, unsigned char *red,
								unsigned char *green, unsigned char *blue, unsigned char type,
								float isoval, int dim[3], float orig[3], float span[3]);

#endif


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

#include <duallib/pcio.h>
#include <assert.h>

size_t getFloat(float *flts, size_t n, FILE *fp)
{
#ifdef SGI
	return fread(flts,4,n,fp);
#else
	unsigned char *pb = new unsigned char[n*4];
	unsigned char *pf = (unsigned char *)flts;
	
	size_t nbytes = fread(pb, 1, 4*n, fp);
	
	if(nbytes == n*4) {
		for(size_t i = 0; i < n; i++) {
			pf[4*i] = pb[4*i+3];
			pf[4*i+1] = pb[4*i+2];
			pf[4*i+2] = pb[4*i+1];
			pf[4*i+3] = pb[4*i];
		}
	}
	delete pb;
	return nbytes;
	
#endif
}

size_t putFloat(float *flts, size_t n, FILE *fp)
{
#ifdef SGI
	return fwrite(flts,4,n,fp);
#else
	unsigned char *pb = new unsigned char[n*4];
	unsigned char *pf = (unsigned char *)flts;
	for(size_t i = 0; i < n; i++) {
		pb[4*i] = pf[4*i+3];
		pb[4*i+1] = pf[4*i+2];
		pb[4*i+2] = pf[4*i+1];
		pb[4*i+3] = pf[4*i];
	}
	
	size_t nbytes = fwrite(pb, 1, 4*n, fp);
	delete pb;
	return nbytes;
#endif
}

size_t getInt(int *Ints, size_t n, FILE *fp)
{
#ifdef SGI
	return fread(Ints,4,n,fp);
#else
	unsigned char *pb = new unsigned char[4*n];
	unsigned char *pf = (unsigned char *)Ints;
	
	int     nbytes = fread(pb, 1, 4*n, fp);
	for(size_t i = 0; i < n; i++) {
		pf[4*i] = pb[4*i+3];
		pf[4*i+1] = pb[4*i+2];
		pf[4*i+2] = pb[4*i+1];
		pf[4*i+3] = pb[4*i];
	}
	delete pb;
	return nbytes;
	
	
#endif
}

size_t getShort(short *shts, size_t n, FILE *fp)
{
#ifdef SGI
		return fread(shts,2,n,fp);
#else 
	unsigned char *pb = new unsigned char[n*2];
	unsigned char *ps = (unsigned char *)shts;
	
	size_t nbytes = fread(pb, sizeof(unsigned char), n*2, fp);
	//swap the byte order
	if(nbytes == n*2) {
		for(size_t i = 0; i < n; i++) {
			ps[2*i] = pb[2*i+1];
			ps[2*i+1] = pb[2*i];
		}
	}
	delete pb;
	return nbytes;	
	
#endif
}

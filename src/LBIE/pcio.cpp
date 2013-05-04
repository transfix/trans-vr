/*
  Copyright 2006 The University of Texas at Austin

        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of LBIE.

  LBIE is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  LBIE is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <LBIE/pcio.h>
#include <assert.h>

namespace LBIE
{

size_t getFloat(float *flts, size_t n, FILE *fp)
{
        unsigned char *pb = new unsigned char[n*4];
        unsigned char *pf = (unsigned char *)flts;
		//float* temp;
		//float tempp;

        size_t nbytes = fread(pb, 1, 4*n, fp);
        //swap the byte order
        if(nbytes == n*4) {
                for(size_t i = 0; i < n; i++) {
                        pf[4*i] = pb[4*i+3];
                        pf[4*i+1] = pb[4*i+2];
                        pf[4*i+2] = pb[4*i+1];
                        pf[4*i+3] = pb[4*i];
		//				temp=(float*)&pf[4*i];
		//				tempp=*temp;

                }
        }
        delete pb;
        return nbytes;
}

size_t putFloat(float *flts, size_t n, FILE *fp)
{
        unsigned char *pb = new unsigned char[n*4];
        unsigned char *pf = (unsigned char *)flts;
		//float* temp;
		//float tempp;

        
        //swap the byte order
        //if(nbytes == n*4) {
                for(size_t i = 0; i < n; i++) {
                        pb[4*i] = pf[4*i+3];
                        pb[4*i+1] = pf[4*i+2];
                        pb[4*i+2] = pf[4*i+1];
                        pb[4*i+3] = pf[4*i];
		//				temp=(float*)&pf[4*i];
		//				tempp=*temp;


                }
        //}
		size_t nbytes = fwrite(pb, 1, 4*n, fp);
        delete pb;
        return nbytes;
}

size_t getInt(int *Ints, size_t n, FILE *fp)
{
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
}

size_t getShort(short *shts, size_t n, FILE *fp)
{
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
}

size_t getUnChar(unsigned char *chars, size_t n, FILE *fp)
{
        unsigned char *pb = new unsigned char[n];
        unsigned char *ps = (unsigned char *)chars;

        size_t nbytes = fread(pb, sizeof(unsigned char), n, fp);
        if(nbytes == n) {
                for(size_t i = 0; i < n; i++) {
                        ps[i] = pb[i];
                }
        }
        delete pb;
        return nbytes;
}

}

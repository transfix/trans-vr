/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of TightCocone.

  TightCocone is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  TightCocone is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/**************************************
 * Zeyun Yu (zeyun@cs.utexas.edu)    *
 * Department of Computer Science    *
 * University of Texas at Austin     *
 **************************************/

/* $Id: segment.h 4741 2011-10-21 21:22:06Z transfix $ */

#include <boost/shared_array.hpp>

namespace TightCocone
{
  typedef struct {
    //float *data;
    //float *tdata;
    boost::shared_array<float> data;
    boost::shared_array<float> tdata;
  } Data_3DS;

  typedef struct {
    int *x;
    int *y;
    int *z;
    float *t;
    int size;
  }MinHeapS;

  typedef struct {
    float x;
    float y;
    float z;
  }VECTOR;

#define max(x, y) ((x>y) ? (x):(y))
#define min(x, y) ((x<y) ? (x):(y))

  //#define IndexVect(i,j,k) ((k)*XDIM*YDIM + (j)*XDIM + (i))

  extern int XDIM;
  extern int YDIM;
  extern int ZDIM;

  /*
    int XDIM;
    int YDIM;
    int ZDIM;
  */
#define TRUE          1
#define FALSE         0
#define MAX_TIME      99999999
#define PIE           3.1415926932
#define WINDOW        2

  extern Data_3DS *dataset;
  extern VECTOR* velocity;
  extern unsigned char *bin_img;
  extern float *ImgGrad;


  inline int IndexVect(int i,int j,int k){
    return ((k)*XDIM*YDIM+ (j)*XDIM + (i));
  }

}

//aj addition
//
/*
  void segment(float, float);
  void Diffuse();
  void read_data(char *input_name);
  void write_data(char *out_seg);
  void GVF_Compute();
*/
// end aj addition
//

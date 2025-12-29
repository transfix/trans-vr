/*
  Copyright 2005-2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolUtils.

  VolUtils is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolUtils is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

/* $Id: img2rawiv+.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <Magick++.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace Magick;

#define SWAP_32(a)                                                           \
  {                                                                          \
    unsigned char tmp[4];                                                    \
    unsigned char *ch;                                                       \
    ch = (unsigned char *)(a);                                               \
    tmp[0] = ch[0];                                                          \
    tmp[1] = ch[1];                                                          \
    tmp[2] = ch[2];                                                          \
    tmp[3] = ch[3];                                                          \
    ch[0] = tmp[3];                                                          \
    ch[1] = tmp[2];                                                          \
    ch[2] = tmp[1];                                                          \
    ch[3] = tmp[0];                                                          \
  }

static inline int big_endian() {
  long one = 1;
  return !(*((char *)(&one)));
}

typedef struct {
  float min[3];
  float max[3];
  unsigned int numVerts;
  unsigned int numCells;
  unsigned int dim[3];
  float origin[3];
  float span[3];
} RawIVHeader;

int main(int argc, char **argv) {
  int width, height, depth, i, j, k;
  Image cur;
  FILE *outvol;
  unsigned char *buf;
  RawIVHeader header;

  if (argc < 4) {
    cout << "Usage: " << argv[0]
         << " <format> <img0> <img1> ... <output rawiv>" << endl;
    cout << "       format... 0 char, 1 short, 2 float" << endl;
    return 0;
  }

  try {
    int type = atoi(argv[1]);

    if (type == 0) {
      cerr << "Datatype: char" << endl;
    } else if (type == 1) {
      cerr << "Datatype: short" << endl;
    } else if (type == 2) {
      cerr << "Datatype: float" << endl;
    } else {
      cerr << "ERROR: bad datatype." << endl;
    }

    // get width and height.  All images should have the same width and height
    // or this program will explode!!!!
    cur.read(argv[2]);
    width = cur.size().width();
    height = cur.size().height();

    cerr << "Image dimensions: " << width << ", " << height << endl;

    // get the depth
    depth = argc - 3;
    cerr << "Volume dimensions: " << width << ", " << height << ", " << depth
         << endl;

    // allocate the slice buffer and open the volume file
    if (type == 0) {
      buf = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    } else if (type == 1) {
      buf = (unsigned char *)malloc(width * height * sizeof(unsigned short));
    } else if (type == 2) {
      buf = (unsigned char *)malloc(width * height * sizeof(float));
    }
    outvol = fopen(argv[argc - 1], "wb");
    if (outvol == NULL) {
      cerr << "Unable to open output file '" << argv[argc - 1] << "'" << endl;
      return 1;
    }

    // fill out the header and write it
    header.min[0] = 0.0;
    header.min[1] = 0.0;
    header.min[2] = 0.0;
    header.max[0] = float(width - 1);
    header.max[1] = float(height - 1);
    header.max[2] = float(depth - 1);
    header.numVerts = width * height * depth;
    header.numCells = (width - 1) * (height - 1) * (depth - 1);
    header.dim[0] = width;
    header.dim[1] = height;
    header.dim[2] = depth;
    header.origin[0] = header.origin[1] = header.origin[2] = 0.0;
    header.span[0] = 1.0;
    header.span[1] = 1.0;
    header.span[2] = 1.0;

    if (!big_endian()) {
      for (i = 0; i < 3; i++)
        SWAP_32(&(header.min[i]));
      for (i = 0; i < 3; i++)
        SWAP_32(&(header.max[i]));
      SWAP_32(&(header.numVerts));
      SWAP_32(&(header.numCells));
      for (i = 0; i < 3; i++)
        SWAP_32(&(header.dim[i]));
      for (i = 0; i < 3; i++)
        SWAP_32(&(header.origin[i]));
      for (i = 0; i < 3; i++)
        SWAP_32(&(header.span[i]));
    }

    fwrite(&header, sizeof(RawIVHeader), 1, outvol);

    // now write out each image slice
    for (k = 0; k < depth; k++) {
      cur.read(Geometry(width, height), argv[k + 2]);
      cur.modifyImage();
      cur.type(GrayscaleType);
      /*
       * We only need 1 value since after grayscale conversion R = G = B
       */

      if (type == 0)
        cur.write(0, 0, width, height, "R", CharPixel, buf);
      else if (type == 1)
        cur.write(0, 0, width, height, "R", ShortPixel, buf);
      else if (type == 2)
        cur.write(0, 0, width, height, "R", FloatPixel, buf);
      /*
        const PixelPacket *pixel_cache = cur.getConstPixels(0,0,width,height);
        for(i=0; i<width; i++)
        for(j=0; j<height; j++)
        {
        const PixelPacket *pixel = pixel_cache+j*width+i;
        if(pixel  == NULL) cout << "i: " << i << ", j: " << j << endl;
        ColorRGB c(*pixel);
        buf[j*width+i] = (unsigned char)(c.green()*255.0);
        }
      */
      if (type == 0)
        fwrite(buf, sizeof(unsigned char), width * height, outvol);
      else if (type == 1)
        fwrite(buf, sizeof(unsigned short), width * height, outvol);
      else if (type == 2)
        fwrite(buf, sizeof(float), width * height, outvol);
    }

    free(buf);
    fclose(outvol);
  } catch (Exception &error_) {
    cout << "Caught exception: " << error_.what() << endl;
    return 1;
  }

  return 0;
}

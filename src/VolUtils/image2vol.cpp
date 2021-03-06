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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: image2vol.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <Magick++.h>
#include <iostream>
#include <VolMagick/VolMagick.h>
#include <cstdio>

using namespace std;
using namespace Magick;

int main(int argc, char **argv)
{
  VolMagick::uint64 width, height, depth, i, j, k;
  Image cur;
  unsigned char *buf;

  if(argc < 3)
    {
      cout << "Usage: " << argv[0] << " <img0> <img1> ... <output volume>" << endl;
      return 0;
    }

  try
    {
#if 0
      // get width and height.  All images should have the same width and height
      // or this program will explode!!!!
      cur.read(argv[1]);
      width = cur.size().width();
      height = cur.size().height();

      cerr << "Image dimensions: " << width << ", " << height << endl;

      // get the depth
      depth = argc-2;
      cerr << "Volume dimensions: " << width << ", " << height << ", " << depth << endl;

      // allocate the slice buffer and open the volume file
      buf = (unsigned char *)malloc(width*height*sizeof(unsigned char));
      outvol = fopen(argv[argc-1],"wb");
      if(outvol == NULL)
	{
	  cerr << "Unable to open output file '" << argv[argc-1] << "'" << endl;
	  return 1;
	}

      // fill out the header and write it
      header.min[0] = 0.0; header.min[1] = 0.0; header.min[2] = 0.0;
      header.max[0] = float(width-1); header.max[1] = float(height-1); header.max[2] = float(depth-1);
      header.numVerts = width * height * depth;
      header.numCells = (width-1)*(height-1)*(depth-1);
      header.dim[0] = width; header.dim[1] = height; header.dim[2] = depth;
      header.origin[0] = header.origin[1] = header.origin[2] = 0.0;
      header.span[0] = 1.0; header.span[1] = 1.0; header.span[2] = 1.0;

      if(!big_endian())
	{
	  for(i=0; i<3; i++) SWAP_32(&(header.min[i]));
	  for(i=0; i<3; i++) SWAP_32(&(header.max[i]));
	  SWAP_32(&(header.numVerts));
	  SWAP_32(&(header.numCells));
	  for(i=0; i<3; i++) SWAP_32(&(header.dim[i]));
	  for(i=0; i<3; i++) SWAP_32(&(header.origin[i]));
	  for(i=0; i<3; i++) SWAP_32(&(header.span[i]));
	}

      fwrite(&header,sizeof(RawIVHeader),1,outvol);

      //now write out each image slice
      for(k=0; k<depth; k++)
	{
	  cur.read(Geometry(width,height),argv[k+1]);
	  cur.modifyImage();
	  cur.type(GrayscaleType);
	  /*
	   * We only need 1 value since after grayscale conversion R = G = B
	   */
	  cur.write(0,0,width,height,"R",CharPixel,buf);

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

	  fwrite(buf,sizeof(unsigned char),width*height,outvol);
	}

      free(buf);
      fclose(outvol);
#endif

      // get width and height.  All images should have the same width and height
      // or this program will explode!!!!
      cur.read(argv[1]);
      width = cur.size().width();
      height = cur.size().height();

      // get the depth
      depth = argc-2;
      cerr << "Volume dimensions: " << width << ", " << height << ", " << depth << endl;

      VolMagick::createVolumeFile(argv[argc-1],
				  VolMagick::BoundingBox(0.0,0.0,0.0,
							 double(width-1),double(height-1),double(depth-1)),
				  VolMagick::Dimension(width,height,depth));

      VolMagick::Volume buf;
      buf.dimension(VolMagick::Dimension(width,height,1));

      //now write out each image slice
      for(k=0; k<depth; k++)
	{
	  cur.read(Geometry(width,height),argv[k+1]);
	  cur.modifyImage();
	  cur.type(GrayscaleType);
	  /*
	   * We only need 1 value since after grayscale conversion R = G = B
	   */
	  cur.write(0,0,width,height,"R",CharPixel,*buf);

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

	  //fwrite(buf,sizeof(unsigned char),width*height,outvol);
	  VolMagick::writeVolumeFile(buf,argv[argc-1],0,0,
				     0,0,k);

	  if(depth>1)
	    fprintf(stderr,"%5.2f %%\r",(double(k)/double(depth-1))*100.0);
	}
      fprintf(stderr,"\nDone!\n");
    }
  catch(std::exception &error_)
    {
      cout << "Caught exception: " << error_.what() << endl;
      return 1;
    }

  return 0;
}

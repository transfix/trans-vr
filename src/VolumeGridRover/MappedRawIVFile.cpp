/*
  Copyright 2005-2008 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeGridRover.

  VolumeGridRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeGridRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <VolumeGridRover/MappedRawIVFile.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USING_QT
#include <qapplication.h>
#include <qprogressdialog.h>
#endif

typedef struct
{
  float min[3];
  float max[3];
  unsigned int numVerts;
  unsigned int numCells;
  unsigned int dim[3];
  float origin[3];
  float span[3];
} RawIVHeader;

MappedRawIVFile::MappedRawIVFile(const char *filename, bool calc_minmax, bool forceLoad)
  : MappedVolumeFile(filename,calc_minmax), m_ForceLoad(forceLoad)
{
  if(m_Valid && !readHeader()) m_Valid = false;
}

MappedRawIVFile::MappedRawIVFile(const char *filename, double mem_usage, bool calc_minmax, bool forceLoad)
  : MappedVolumeFile(filename,mem_usage,calc_minmax), m_ForceLoad(forceLoad)
{
  if(m_Valid && !readHeader()) m_Valid = false;
}
 
MappedRawIVFile::~MappedRawIVFile()
{
  if(m_Variables && m_Variables[0]) delete m_Variables[0];
  if(m_Variables) delete [] m_Variables;
}

bool MappedRawIVFile::readHeader()
{
  int rawiv_type_sizes[] = { 1, 2, 4, 4, 8 };
#ifdef DEBUG
  char *rawiv_type_strings[] = { "unsigned char", "unsigned short", "unsigned int/long", "float", "double" };
#endif
  unsigned int i,a,b,c;
  Variable::VariableType vartype;
  RawIVHeader header;
  lfmap_ptr_t head_ptr;

  head_ptr = lfmap_ptr(m_LFMappedVolumeFile,0,sizeof(RawIVHeader));
  if(head_ptr == NULL)
    {
      fprintf(stderr,"MappedRawIVFile::readHeader(): Error: Truncated header, invalid RawIV file.\n");
      return false;
    }
  memcpy(&header,head_ptr,sizeof(RawIVHeader));

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
  
  m_XDim = header.dim[0];
  m_YDim = header.dim[1];
  m_ZDim = header.dim[2];

  /* error checking */
  if(m_XDim == 0 || m_YDim == 0 || m_ZDim == 0)
    {
      fprintf(stderr,"MappedRawIVFile::readHeader(): Invalid volume dimensions.\n");
      return false;
    }
  if(header.numVerts != (m_XDim * m_YDim * m_ZDim))
    {
      fprintf(stderr,"MappedRawIVFile::readHeader(): Number of vertices does not match calculated value, possible header corruption\n");
      if(!m_ForceLoad) return false;
      fprintf(stderr,"MappedRawIVFile::readHeader(): Ignoring possible error.\n");
    }
  if(header.numCells != ((m_XDim-1)*(m_YDim-1)*(m_ZDim-1)))
    {
      fprintf(stderr,"MappedRawIVFile::readHeader(): Number of cells does not match calculated value, possible header corruption\n");
      if(!m_ForceLoad) return false;
      fprintf(stderr,"MappedRawIVFile::readHeader(): Ignoring possible error.\n");
    }
  for(i=0; i<3; i++)
    if(fabs(header.span[i] - ((header.max[i] - header.min[i])/(header.dim[i] - 1))) > 0.01f)
      {
	fprintf(stderr,"MappedRawIVFile::readHeader(): Span value (%f) does not match calculation (%f), possible header corruption\n",
		header.span[i],((header.max[i] - header.min[i])/(header.dim[i] - 1)));
	if(!m_ForceLoad) return false;
	fprintf(stderr,"MappedRawIVFile::readHeader(): Ignoring possible error.\n");
      }
  m_NumVariables = 1;
  m_NumTimesteps = 1;
  m_Variables = new Variable*[1];
  switch((m_Filesize-68)/(m_XDim*m_YDim*m_ZDim))
    {
    case 1: vartype = Variable::UCHAR; break;
    case 2: vartype = Variable::USHORT; break;
    case 4: vartype = Variable::FLOAT; break;
    case 8: vartype = Variable::DOUBLE; break;
    default: 
      fprintf(stderr,"MappedRawIVFile::readHeader(): Unknown data type, assuming unsigned char.\n");
      vartype = Variable::UCHAR;
      break;
    }

  if(m_Filesize != (m_XDim*m_YDim*m_ZDim*rawiv_type_sizes[vartype] + 68))
    {
      fprintf(stderr,"MappedRawIVFile::readHeader(): Volume dimensions do not match filesize. (file size: %lld, calculated size: %lld)\n",
	      m_Filesize,lfmap_uint64_t(m_XDim*m_YDim*m_ZDim*rawiv_type_sizes[vartype] + 68));
      return false;
    }

  m_Variables[0] = new Variable(this,68,"No Name",vartype,!big_endian());
  
  m_XSpan = header.span[0];
  m_YSpan = header.span[1];
  m_ZSpan = header.span[2];
  m_TSpan = 1.0f;

#ifdef DEBUG
  printf("MappedRawIVFile::readHeader(): XDim: %lld, YDim: %lld, ZDim: %lld\n",m_XDim,m_YDim,m_ZDim);
  printf("MappedRawIVFile::readHeader(): XSpan: %f, YSpan: %f, ZSpan: %f, TSpan: %f\n",m_XSpan,m_YSpan,m_ZSpan,m_TSpan);
  printf("MappedRawIVFile::readHeader(): Num Variables: 1, Num Timesteps: 1\n");
  printf("MappedRawIVFile::readHeader(): Variable Type: '%s'\n",rawiv_type_strings[vartype]);
#endif

  /* 
	get the min/max value.
	Here we use an extension to the rawiv format if it's available.  Since the origin values are not really
	used, we can use them to encode the min and max density values.  If the origin X value has the bytes
	0xBAADBEEF, then origin y is the minimum density value and origin z is the maximum.
  */
  unsigned int *tmp_int = (unsigned int*)(&(header.origin[0])); /* use a pointer so the compiler doesn't try to convert the float value to int */
  if(*tmp_int == 0xBAADBEEF)
  {
	printf("MappedRawIVFile::readHeader(): Using rawiv extension: Min and max density values encoded in origin y and origin z header bytes\n");
	m_Variables[0]->m_Min = header.origin[1];
	m_Variables[0]->m_Max = header.origin[2];
  }
  else if(vartype == Variable::UCHAR || !m_CalcMinMax)
    {
      m_Variables[0]->m_Min = 0.0;
      m_Variables[0]->m_Max = 255.0;
    }
  else
    {
      lfmap_ptr_t slice = (lfmap_ptr_t)malloc(m_XDim*m_YDim*rawiv_type_sizes[vartype]);
#ifdef USING_QT
      QProgressDialog progress("Calculating variable min/max for variable 0, timestep 0","Abort",m_ZDim,NULL,"progress",true);
      progress.show();
#endif

      m_Variables[0]->m_Min = m_Variables[0]->m_Max = m_Variables[0]->get(0,0,0);
      for(c=0; c<m_ZDim; c++)
	{
	  double val;
	  
	  m_Variables[0]->get(0,0,c,m_XDim,m_YDim,1,slice); /* get a slice at a time because it's much faster */

#define GETMIN(_vartype_)							        \
	  {								                \
	    for(a=0; a<m_XDim; a++)					                \
	      for(b=0; b<m_YDim; b++)					                \
		{							                \
		  val = double(*((_vartype_ *)(slice+(a+m_XDim*b)*sizeof(_vartype_)))); \
		  if(m_Variables[0]->m_Min > val)			                \
		    m_Variables[0]->m_Min = val;			                \
		  else if(m_Variables[0]->m_Max < val)		                        \
		    m_Variables[0]->m_Max = val;			                \
		}							                \
	  }

	  switch(vartype)
	    {
	    case Variable::UCHAR:  GETMIN(unsigned char);  break;
	    case Variable::USHORT: GETMIN(unsigned short); break;
	    case Variable::UINT:   GETMIN(unsigned int);   break;
	    case Variable::FLOAT:  GETMIN(float);          break;
	    case Variable::DOUBLE: GETMIN(double);         break;
	    }

#undef GETMIN
	
#ifdef DEBUG	
	  fprintf(stderr,"%5.2f %%\r",(((float)c)/((float)((int)(m_ZDim-1))))*100.0);
#endif

#ifdef USING_QT
	  progress.setProgress(c);
	  qApp->processEvents();
	  if(progress.wasCanceled())
	    return false;
#endif
	}
#ifdef DEBUG
      printf("\n");
#endif

#ifdef USING_QT
      progress.setProgress(m_ZDim);
#endif
      free(slice);
    }

#ifdef DEBUG
  printf("MappedRawIVFile::readHeader(): Variable 0, Timestep 0: min: %f, max: %f\n",m_Variables[0]->m_Min,m_Variables[0]->m_Max);
#endif
  
  return true;
}

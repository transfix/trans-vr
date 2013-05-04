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

/* $Id: MappedRawVFile.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <VolUtils/MappedRawVFile.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USING_QT
#include <qapplication.h>
#include <qprogressdialog.h>
#include <qstring.h>
#endif

typedef struct
{
  unsigned int magic;
  unsigned int dim[3];
  unsigned int numTimesteps;
  unsigned int numVariables;
  float min[4];
  float max[4];
  /* variable records come next */
} RawVHeader;

typedef struct
{
  unsigned char varType;
  char varName[64];
} VariableRecord;

MappedRawVFile::MappedRawVFile(const char *filename, bool calc_minmax, bool forceLoad)
 : MappedVolumeFile(filename,calc_minmax), m_ForceLoad(forceLoad)
{
  if(m_Valid && !readHeader()) m_Valid = false;
}

MappedRawVFile::MappedRawVFile(const char *filename, double mem_usage, bool calc_minmax, bool forceLoad)
  : MappedVolumeFile(filename,mem_usage,calc_minmax), m_ForceLoad(forceLoad)
{
  if(m_Valid && !readHeader()) m_Valid = false;
}

MappedRawVFile::~MappedRawVFile()
{
  for(unsigned int i=0; i<m_NumVariables*m_NumTimesteps; i++)
    delete m_Variables[i];
  if(m_Variables) delete [] m_Variables;
}

bool MappedRawVFile::readHeader()
{
  Variable::VariableType rawv_type_conv[] = { Variable::UCHAR, Variable::UCHAR, Variable::USHORT, Variable::UINT, Variable::FLOAT, Variable::DOUBLE };
  int rawv_type_sizes[] = { 0, 1, 2, 4, 4, 8 };
  char *rawv_type_strings[] = { NULL, "unsigned char", "unsigned short", "unsigned int/long", "float", "double" };
  unsigned int i,j,a,b,c;
  VariableRecord *var_records;
  RawVHeader header;
  lfmap_ptr_t head_ptr, var_rec_ptr;

  head_ptr = lfmap_ptr(m_LFMappedVolumeFile,0,sizeof(RawVHeader));
  if(head_ptr == NULL)
    {
      fprintf(stderr,"MappedRawVFile::readHeader(): Truncated header, invalid RawV file.\n");
      return false;
    }

  memcpy(&header,head_ptr,sizeof(RawVHeader));
  
  if(!big_endian())
    {
      SWAP_32(&(header.magic));
      for(i=0; i<3; i++) SWAP_32(&(header.dim[i]));
      SWAP_32(&(header.numTimesteps));
      SWAP_32(&(header.numVariables));
      for(i=0; i<4; i++) SWAP_32(&(header.min[i]));
      for(i=0; i<4; i++) SWAP_32(&(header.max[i]));
    }
  
  /* initial error check */
  if(header.magic != 0xBAADBEEF)
    {
      fprintf(stderr,"MappedRawVFile::readHeader(): Error: Magic number not present in file.\n");
      return false;
    }
  
  /* variable initialization */
  m_XDim = header.dim[0];
  m_YDim = header.dim[1];
  m_ZDim = header.dim[2];
  m_XSpan = (header.max[0] - header.min[0])/(header.dim[0] - 1);
  m_YSpan = (header.max[1] - header.min[1])/(header.dim[1] - 1);
  m_ZSpan = (header.max[2] - header.min[2])/(header.dim[2] - 1);
  m_TSpan = (header.max[3] - header.min[3])/header.numTimesteps;
#ifdef DEBUG
  printf("MappedRawVFile::readHeader(): XDim: %lld, YDim: %lld, ZDim: %lld\n",m_XDim,m_YDim,m_ZDim);
  printf("MappedRawVFile::readHeader(): XSpan: %f, YSpan: %f, ZSpan: %f, TSpan: %f\n",m_XSpan,m_YSpan,m_ZSpan,m_TSpan);
#endif
  m_NumVariables = header.numVariables;
  m_NumTimesteps = header.numTimesteps;
#ifdef DEBUG
  printf("MappedRawVFile::readHeader(): Num Variables: %d, Num Timesteps: %d\n",m_NumVariables,m_NumTimesteps);
#endif
  
  /* error checking */
  lfmap_uint64_t dataBytes=0;
  if(sizeof(RawVHeader)+sizeof(VariableRecord)*m_NumVariables>=m_Filesize)
  {
    fprintf(stderr,"MappedRawVFile::readHeader(): Error: Incorrect filesize.\n");
    return false;
  }
  if(m_NumVariables == 0)
  {
    fprintf(stderr,"MappedRawVFile::readHeader(): Error: Number of variables == 0.\n");
    return false;
  }

  /* make a copy of the variable records in the case that the pointer changes due to
     an lfmap remapping */
  var_records = (VariableRecord*)malloc(m_NumVariables*sizeof(VariableRecord));
  var_rec_ptr = lfmap_ptr(m_LFMappedVolumeFile,sizeof(RawVHeader),sizeof(VariableRecord)*m_NumVariables);
  if(var_rec_ptr == NULL)
    {
      fprintf(stderr,"MappedRawVFile::readHeader(): Error: Variable records truncated, not a RawV file.\n");
      free(var_records);
      return false;
    }
  memcpy(var_records,var_rec_ptr,m_NumVariables*sizeof(VariableRecord));

  for(i=0; i<m_NumVariables; i++)
  {
#ifdef DEBUG
    printf("MappedRawVFile::readHeader(): Checking variable record for variable '%d'.\n",i);
#endif
    /* check for null byte in variable name */
    for(j=0; j<64; j++)
      if(var_records[i].varName[j] == '\0') break;
    if(j==64)
      {
	fprintf(stderr,"MappedRawVFile::readHeader(): Error: Non null terminated variable name for variable '%d'\n",i);
	free(var_records);
	return false;
      }
	 
    if(var_records[i].varType > 5)
      {
	fprintf(stderr,"MappedRawVFile::readHeader(): Illegal variable type '%d'.\n",var_records[i].varType);
	free(var_records);
	return false;
      }
    dataBytes += m_XDim*m_YDim*m_ZDim*rawv_type_sizes[var_records[i].varType]*m_NumTimesteps;
#ifdef DEBUG
    printf("MappedRawVFile::readHeader(): Variable record for variable '%d' ('%s' of type '%s') correct.\n",
	   i,var_records[i].varName,rawv_type_strings[var_records[i].varType]);
#endif
  }
  if(sizeof(RawVHeader)+sizeof(VariableRecord)*m_NumVariables+dataBytes != m_Filesize)
    {
      fprintf(stderr,"MappedRawVFile::readHeader(): File size does not match header info.\n");
      free(var_records);
      return false;
    }
  
  m_Variables = new Variable*[m_NumVariables*m_NumTimesteps];
  for(i=0; i<m_NumVariables; i++)
    for(j=0; j<m_NumTimesteps; j++)
      {
	int index = i+m_NumVariables*j;
	lfmap_uint64_t single_length = m_XDim*m_YDim*m_ZDim*rawv_type_sizes[var_records[i].varType];
	lfmap_uint64_t var_start=0;
	for(unsigned int d=0; d<i; d++) /* count the number of bytes to the start of this variable */
	  var_start += m_XDim*m_YDim*m_ZDim*rawv_type_sizes[var_records[d].varType]*m_NumTimesteps;
#ifdef DEBUG
	printf("MappedRawVFile::readHeader(): Reading variable '%s' of type '%s' (timestep %d)\n",var_records[i].varName,rawv_type_strings[var_records[i].varType],j);
#endif
	m_Variables[index] = new Variable(this,(sizeof(RawVHeader)+sizeof(VariableRecord)*m_NumVariables)+var_start+single_length*j,var_records[i].varName,rawv_type_conv[var_records[i].varType],!big_endian());
	/* get the min/max value */
	if(rawv_type_conv[var_records[i].varType] == Variable::UCHAR || !m_CalcMinMax)
	  {
	    m_Variables[index]->m_Min = 0;
	    m_Variables[index]->m_Max = 255;
	  }
	else
	  {
	    lfmap_ptr_t slice = (lfmap_ptr_t)malloc(m_XDim*m_YDim*rawv_type_sizes[var_records[i].varType]);
#ifdef USING_QT
	    QProgressDialog progress(QString("Calculating variable min/max for variable %1, timestep %2").arg(i).arg(j),
				     "Abort",m_ZDim,NULL,"progress",true);
	    progress.show();
#endif

	    m_Variables[index]->m_Min = m_Variables[index]->m_Max = m_Variables[index]->get(0,0,0);
	    for(c=0; c<m_ZDim; c++)
	      {
		double val;

		m_Variables[index]->get(0,0,c,m_XDim,m_YDim,1,slice); /* get a slice at a time because it's much faster */

#define GETMIN(vartype)                                                                                      \
		{                                                                                            \
		  for(a=0; a<m_XDim; a++)                                                                    \
		    for(b=0; b<m_YDim; b++)                                                                  \
		      {                                                                                      \
			val = double(*((vartype *)(slice+(a+m_XDim*b)*sizeof(vartype))));                    \
			if(m_Variables[index]->m_Min > val)                                                  \
			  m_Variables[index]->m_Min = val;                                                   \
			else if(m_Variables[index]->m_Max < val)                                             \
			  m_Variables[index]->m_Max = val;                                                   \
		      }                                                                                      \
		}

		switch(rawv_type_conv[var_records[i].varType])
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
	printf("MappedRawVFile::readHeader(): Variable %d, Timestep %d: min: %f, max: %f\n",i,j,m_Variables[index]->m_Min,m_Variables[index]->m_Max);
#endif
      }

  free(var_records);
  return true;
}

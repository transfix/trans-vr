/*
  Copyright 2007-2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolMagick.

  VolMagick is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolMagick is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* readvtk/writevtk:
 * Copyright (c) The Technical University of Denmark
 * Author: Ojaswa Sharma
 * E-mail: os@imm.dtu.dk
 * File: vtkIO.h
 * .vtk I/O
 */

/* $Id: VTK_IO.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <errno.h>

#include <boost/format.hpp>

#include <VolMagick/VolMagick.h>
#include <VolMagick/VTK_IO.h>
#include <VolMagick/endians.h>

#include <CVC/App.h>

#ifdef __WINDOWS__ 
#define SNPRINTF _snprintf
#define FSEEK fseek
#else
#define SNPRINTF snprintf
#define FSEEK fseeko
#endif

namespace VolMagick
{
#define LINE_LENGTH 256
  typedef float FLOAT; // doubles are supported by VTK format, but here we just want to read as float.

  struct DataInfo
  {
    unsigned int n[3]; // Number of elements in each dimension
    unsigned int n_input[3]; // NUmber of elements in the input volume
    unsigned long nTotal; // Total number of voxels = n[0]*n[1]*n[3]
    char dataType[256]; // data type of values. -ve value indicated signed data
    bool volumeCovered; //cover the input volume with a one voxel layer around - binary flag.
  };

  FLOAT* readvtk(const char * filename, DataInfo& di, unsigned int subvol_dim, bool coverVolume = false);
  void writevtk(const char * filename, const DataInfo& di, const FLOAT* data, float value_increment);

  static inline void geterrstr(int errnum, char *strerrbuf, size_t buflen)
  {
#ifdef HAVE_STRERROR_R
    strerror_r(errnum,strerrbuf,buflen);
#else
    SNPRINTF(strerrbuf,buflen,"%s",strerror(errnum)); /* hopefully this is thread-safe on the target system! */
#endif
  }

  // -------
  // readvtk
  // -------
  // Purpose:
  //   Reads a VTK file and returns a pointer to the data. The binary file is in Big-Endian format.
  // ---- Change History ----
  // ??/??/2008 -- Ojaswa S. -- Initial implementation.
  FLOAT* readvtk(const char * filename, DataInfo& di, unsigned int subvol_dim, bool coverVolume)
  {
    using namespace std;
    using namespace boost;

    char buf[256];
    // Read file supplied as an argument... 
    fprintf(stderr, "Reading datafile %s...", filename);
    FILE *fd = fopen(filename, "rb");
    if(fd == NULL)
      {
	geterrstr(errno,buf,256);
	string errStr = "Error opening file '" + string(filename) + "': " + string(buf);
	throw ReadError(errStr);
      }

    char version[LINE_LENGTH];
    char comments[LINE_LENGTH];
    char format_[LINE_LENGTH]; //"BINARY", "ASCII"
    char type[LINE_LENGTH]; //"DATASET STRUCTURED_POINTS"
    char dimensions[LINE_LENGTH]; //"DIMENSIONS NX NY NZ"
    char origin[LINE_LENGTH]; //"ORIGIN OX OY OZ"
    char spacing[LINE_LENGTH]; //"SPACING SX SY SZ"
    char num_pts[LINE_LENGTH]; //"POINT_DATA NN", NN = NX*NY*NZ
    char data_type[LINE_LENGTH]; //"SCALARS name data_type", e.g. "image_data unsigned_short"
    char lookup_t[LINE_LENGTH]; //"LOOKUP_TABLE default"
  
    fgets(version, LINE_LENGTH, fd);
    fgets(comments, LINE_LENGTH, fd);
    fgets(format_, LINE_LENGTH, fd);
    fgets(type, LINE_LENGTH, fd);
    fgets(dimensions, LINE_LENGTH, fd);
    fgets(origin, LINE_LENGTH, fd);
    fgets(spacing, LINE_LENGTH, fd);
    fgets(num_pts, LINE_LENGTH, fd);
    fgets(data_type, LINE_LENGTH, fd);
    fgets(lookup_t, LINE_LENGTH, fd);

    if(strncmp(format_, "BINARY", 6) != 0)
      {
	string errStr = "Error reading file '" + string(filename) + "': Only binary .vtk files are supported.";
	fclose(fd);
	throw ReadError(errStr);
      }

    char _type[256];
    sscanf(type, "%*s %s", _type);
    if(strncmp(_type, "STRUCTURED_POINTS", 17) != 0)
      {
	string errStr = "Error reading file '" + string(filename) + "': Only structured points are supported.";
	fclose(fd); 
	throw ReadError(errStr);
      }

    char _dimstr[256];
    sscanf(dimensions, "%s %u %u %u", _dimstr, &(di.n_input[0]), &(di.n_input[1]), &(di.n_input[2]));
    if(strncmp(_dimstr, "DIMENSIONS", 10) != 0)
      {
	string errStr = "Error reading file '" + string(filename) + "': Corrupt header for dimension.";
	fclose(fd);
	throw ReadError(errStr);
      }

    di.volumeCovered  = coverVolume;
    if(coverVolume)
      {
	di.n[0] = di.n_input[0] + 2;
	di.n[1] = di.n_input[1] + 2;
	di.n[2] = di.n_input[2] + 2;
      }
    //pad to match subvolume multiples
    subvol_dim -=2; // Since we have a 2 voxel overlap between subvolumes.
    unsigned int n = ((di.n[0] % subvol_dim) != 0)? (di.n[0] / subvol_dim + 1):(di.n[0] / subvol_dim);
    di.n[0] = 2 + n*subvol_dim;
    n = ((di.n[1] % subvol_dim) != 0)? (di.n[1] / subvol_dim + 1):(di.n[1] / subvol_dim);
    di.n[1] = 2 + n*subvol_dim;
    n = ((di.n[2] % subvol_dim) != 0)? (di.n[2] / subvol_dim + 1):(di.n[2] / subvol_dim);
    di.n[2] = 2 + n*subvol_dim;

    char _numptsstr[256];
    sscanf(num_pts, "%s %lu", _numptsstr, &(di.nTotal));
    if(strncmp(_numptsstr, "POINT_DATA", 10) != 0)
      {
	string errStr = "Error reading file '" + string(filename) + "': Corrupt header for data type.";
	fclose(fd);
	throw ReadError(errStr);
      }
  
    char _datastr[256], _dataname[256];
    sscanf(data_type, "%s %s %s", _datastr, _dataname, di.dataType);
    if(strncmp(_datastr, "SCALARS", 7) != 0 || strncmp(_dataname, "image_data", 10) !=0)
      {
	string errStr = "Error reading file '" + 
	  string(filename) + "': Only scalar data of type \"image_data\" of type \"unsigned_short\" is supported.";
	fclose(fd); 
	throw ReadError(errStr);
      } 

    fprintf(stderr, "\nInput volume size:  %d x %d x %d, %s\n", di.n_input[0], di.n_input[1], di.n_input[2], di.dataType);
    fprintf(stderr, "\nPadded volume size: %d x %d x %d\n", di.n[0], di.n[1], di.n[2]);
    //fprintf(stderr, "%s.\n%s.\n%s.\n%s.\n%s.\n%s.\n%s.\n%s.\n%s.\n%s.\n", version, comments, format, type, dimensions, origin, spacing, num_pts, data_type, lookup_t);

    //Allocate memory
    unsigned long nMem = (di.n[0]*di.n[1]*di.n[2]);
    unsigned byte_size = 0;
    if(strncmp(di.dataType, "unsigned_char", 13) == 0)
      byte_size = sizeof(unsigned char);
    else if(strncmp(di.dataType, "char", 4) == 0)
      byte_size = sizeof(char);
    else if(strncmp(di.dataType, "unsigned_short", 14) == 0)
      byte_size = sizeof(unsigned short);
    else if(strncmp(di.dataType, "short", 5) == 0)
      byte_size = sizeof(short);
    else if(strncmp(di.dataType, "unsigned_int", 12) == 0)
      byte_size = sizeof(unsigned int);
    else if(strncmp(di.dataType, "int", 3) == 0)
      byte_size = sizeof(int);
    else if(strncmp(di.dataType, "float", 5) == 0)
      byte_size = sizeof(float);
    else if(strncmp(di.dataType, "double", 6) == 0)
      byte_size = sizeof(double);
    // N.B.: our data currently is 16 bit unsigned int (unsigned short). We therefore do not worry about other types!!!
    if(strncmp(di.dataType, "unsigned_short", 14) != 0)
      {
	string errStr = "Error reading file '" + 
	  string(filename) + "': Only 16 bit unsignd integers are supported at the moment.";
	fclose(fd);
	throw ReadError(errStr);
      }

    FLOAT* dataPtr = (FLOAT*)calloc(nMem, sizeof(FLOAT)); // initialize mem to zero - so that padded part is set to zero!
    if(dataPtr == NULL)
      {
	string errStr(str(format("Error reading file '%1%': Could not allocate %2% bytes of memory!") 
			  % string(filename)
			  % (nMem*sizeof(FLOAT))));
	fclose(fd); 
	throw ReadError(errStr);
      }

    unsigned short dataVal;
    register unsigned int _r, _c, _d;
    FLOAT *ptr = NULL;
    for(_d = 0; _d < di.n_input[2]; _d++)
      for(_c = 0; _c < di.n_input[1]; _c++)
	for(_r = 0; _r < di.n_input[0]; _r++)
	  {
	    if(feof(fd) > 0)
	      {
		string errStr = "Error reading file '" + 
		  string(filename) + "': Insufficient elements in the file.";
		free(dataPtr);
		fclose(fd);
		throw ReadError(errStr);
	      }  
	    fread(&dataVal, byte_size, 1, fd);
	    //dataVal = ntohs(dataVal);
	    if(!big_endian())
	      SWAP_16(&dataVal);
	    ptr = coverVolume?(dataPtr + _r + 1 + (_c + 1)*di.n[0] + (_d + 1)*di.n[0]*di.n[1]):(dataPtr + _r + _c*di.n[0] + _d*di.n[0]*di.n[1]);
	    *ptr = (FLOAT)(dataVal);
	  }

    fclose(fd);
    fprintf(stderr, "Done.\n");
    return dataPtr; //return pointer to data
  }

  // --------
  // writevtk
  // --------
  // Purpose:
  //   Writes a VTK file.
  // ---- Change History ----
  // ??/??/2008 -- Ojaswa S. -- Initial implementation.
  void writevtk(const char * filename, const DataInfo& di, const FLOAT* dataPtr, float value_increment)
  {
    using namespace std;
    using namespace boost;

    char buf[256];
    // write file supplied as an argument... 
    //bool stripCover = di.volumeCovered;
    fprintf(stderr, "Writing datafile %s...", filename);
    FILE *fd = fopen(filename, "wb");
    if(fd == NULL)
      {
	geterrstr(errno,buf,256);
	string errStr = "Error opening file '" + string(filename) + "': " + string(buf);
	throw WriteError(errStr);
      }
    char str[256];
    unsigned long nMem = di.n[0]*di.n[1]*di.n[2];
    sprintf(str, "# vtk DataFile Version 3.0\n");
    fputs(str, fd);
    sprintf(str, "created by levelSet (Ojaswa Sharma). Zero level: %f\n", value_increment);
    fputs(str, fd);
    sprintf(str, "BINARY\n");
    fputs(str, fd);  
    sprintf(str, "DATASET STRUCTURED_POINTS\n");
    fputs(str, fd);  
    sprintf(str, "DIMENSIONS %d %d %d\n", di.n[0], di.n[1], di.n[2]);
    fputs(str, fd);  
    sprintf(str, "ORIGIN 0.000000 0.000000 0.000000\n");
    fputs(str, fd);  
    sprintf(str, "SPACING 1.000000 1.000000 1.000000\n");
    fputs(str, fd);  
    sprintf(str, "POINT_DATA %lu\n", nMem);
    fputs(str, fd);  
    sprintf(str, "SCALARS image_data unsigned_short\n");
    fputs(str, fd);  
    sprintf(str, "LOOKUP_TABLE default\n");
    fputs(str, fd);  
    unsigned short dataval;
    FLOAT datavalf;
    register unsigned int _r, _c, _d;
    const FLOAT *ptr = NULL;
    for(_d = 0; _d < di.n[2]; _d++)
      for(_c = 0; _c < di.n[1]; _c++)
	for(_r = 0; _r < di.n[0]; _r++)
	  {
	    ptr = dataPtr + _r + _c*di.n[0] + _d*di.n[0]*di.n[1];
	    datavalf = *ptr + value_increment;
	    if(datavalf < 0.0) datavalf = 0.0;    
        //dataval = (unsigned short)(round(datavalf));
        dataval = (unsigned short)(floor(datavalf+0.5));
	    //dataval = htons(dataval);
	    if(!big_endian())
	      SWAP_16(&dataval);
	    if(fwrite(&dataval, sizeof(unsigned short), 1, fd)!=1)
	      {
		geterrstr(errno,buf,256);
		std::string errStr = "Error writing volume data to file '" + 
		  string(filename) + "': " + buf;
		fclose(fd);
		throw WriteError(errStr);
	      }
	  }
    fclose(fd);
    fprintf(stderr, "Done\n");
  }

  // --------------
  // VTK_IO::VTK_IO
  // --------------
  // Purpose:
  //   Initializes the extension list and id.
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation.
  VTK_IO::VTK_IO()
    : _id("VTK_IO : v1.0")
  {
    _extensions.push_back(".vtk");
  }

  // ----------
  // VTK_IO::id
  // ----------
  // Purpose:
  //   Returns a string that identifies this VolumeFile_IO object.  This should
  //   be unique, but is freeform.
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation.
  const std::string& VTK_IO::id() const
  {
    return _id;
  }

  // ------------------
  // VTK_IO::extensions
  // ------------------
  // Purpose:
  //   Returns a list of extensions that this VolumeFile_IO object supports.
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation.
  const VolumeFile_IO::ExtensionList& VTK_IO::extensions() const
  {
    return _extensions;
  }

  // -------------------------
  // VTK_IO::getVolumeFileInfo
  // -------------------------
  // Purpose:
  //   Writes to a structure containing all info that VolMagick needs
  //   from a volume file.
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation.
  void VTK_IO::getVolumeFileInfo(VolumeFileInfo::Data& /*data*/,
				 const std::string& /*filename*/) const
  {
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);
    throw ReadError("Reading VTK files doesn't work yet!");
  }

  // ----------------------
  // VTK_IO::readVolumeFile
  // ----------------------
  // Purpose:
  //   Writes to a Volume object after reading from a volume file.
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation.
  void VTK_IO::readVolumeFile(Volume& /*vol*/,
			      const std::string& /*filename*/, 
			      unsigned int /*var*/, unsigned int /*time*/,
			      uint64 /*off_x*/, uint64 /*off_y*/, uint64 /*off_z*/,
			      const Dimension& /*subvoldim*/) const
  {
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);
    throw ReadError("Reading VTK files doesn't work yet!");
  }

  // ------------------------
  // VTK_IO::createVolumeFile
  // ------------------------
  // Purpose:
  //   Creates an empty volume file to be later filled in by writeVolumeFile
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation.
  void VTK_IO::createVolumeFile(const std::string& /*filename*/,
				const BoundingBox& /*boundingBox*/,
				const Dimension& /*dimension*/,
				const std::vector<VoxelType>& /*voxelTypes*/,
				unsigned int /*numVariables*/, unsigned int /*numTimesteps*/,
				double /*min_time*/, double /*max_time*/) const
  {
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);
    throw WriteError("Writing VTK files doesn't work yet!");
  }

  // -----------------------
  // VTK_IO::writeVolumeFile
  // -----------------------
  // Purpose:
  //   Writes the volume contained in wvol to the specified volume file. Should create
  //   a volume file if the filename provided doesn't exist.  Else it will simply
  //   write data to the existing file.  A common user error arises when you try to
  //   write over an existing volume file using this function for unrelated volumes.
  //   If what you desire is to overwrite an existing volume file, first run
  //   createVolumeFile to replace the volume file.
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation.
  void VTK_IO::writeVolumeFile(const Volume& /*wvol*/, 
			       const std::string& /*filename*/,
			       unsigned int /*var*/, unsigned int /*time*/,
			       uint64 /*off_x*/, uint64 /*off_y*/, uint64 /*off_z*/) const
  {
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);
    throw WriteError("Writing VTK files doesn't work yet!");
  }
}

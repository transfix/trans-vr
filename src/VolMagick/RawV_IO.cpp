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

/* $Id: RawV_IO.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <VolMagick/VolMagick.h>
#include <VolMagick/endians.h>
#include <VolMagick/RawV_IO.h>

#include <CVC/App.h>

#ifdef __WINDOWS__ 
#define SNPRINTF _snprintf
#define FSEEK fseek
#else
#define SNPRINTF snprintf
#define FSEEK fseeko
#endif

static inline void geterrstr(int errnum, char *strerrbuf, size_t buflen)
{
#ifdef HAVE_STRERROR_R
  strerror_r(errnum,strerrbuf,buflen);
#else
  SNPRINTF(strerrbuf,buflen,"%s",strerror(errnum)); /* hopefully this is thread-safe on the target system! */
#endif
}

typedef struct header_t
{
  unsigned int magic;
  unsigned int dim[3];
  unsigned int numTimesteps;
  unsigned int numVariables;
  float min[4];
  float max[4];
  /* variable records come next */
} header_t;

typedef struct var_record_t
{
  unsigned char varType;
  char varName[64];
} var_record_t;

namespace VolMagick
{
  // ----------------
  // RawV_IO::RawV_IO
  // ----------------
  // Purpose:
  //   Initializes the extension list and id.
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation
  RawV_IO::RawV_IO()
    : _id("RawV_IO : v1.0")
  {
    _extensions.push_back(".rawv");
  }

  // -----------
  // RawV_IO::id
  // -----------
  // Purpose:
  //   Returns a string that identifies this VolumeFile_IO object.  This should
  //   be unique, but is freeform.
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation
  const std::string& RawV_IO::id() const
  {
    return _id;
  }

  // -------------------
  // RawV_IO::extensions
  // -------------------
  // Purpose:
  //   Returns a list of extensions that this VolumeFile_IO object supports.
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation
  const VolumeFile_IO::ExtensionList& RawV_IO::extensions() const
  {
    return _extensions;
  }

  // --------------------------
  // RawV_IO::getVolumeFileInfo
  // --------------------------
  // Purpose:
  //   Writes to a structure containing all info that VolMagick needs
  //   from a volume file.
  // ---- Change History ----
  // ??/??/2007 -- Joe R. -- Initial implementation.
  // 11/20/2009 -- Joe R. -- Converted to a VolumeFile_IO class
  void RawV_IO::getVolumeFileInfo(VolumeFileInfo::Data& data,
				  const std::string& filename) const
  {
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);

    char buf[256];
    VoxelType rawv_type_conv[] = { CVC::UChar, CVC::UChar, CVC::UShort, 
                                   CVC::UInt, CVC::Float, CVC::Double };
    uint64 rawv_type_sizes[] = { 0, 1, 2, 4, 4, 8 };
    
    header_t header;
    var_record_t *var_records;

    FILE *input;
    struct stat s;
    uint64 dataBytes;
    unsigned int i,j;

    memset(buf,0,256);

    if((input = fopen(filename.c_str(),"rb")) == NULL)
      {
	geterrstr(errno,buf,256);
        std::string errStr = "Error opening file '" + filename + "': " + buf;
	throw ReadError(errStr);
      }

    if(fread(&header, sizeof(header_t), 1, input) != 1)
      {
	geterrstr(errno,buf,256);
        std::string errStr = "Error reading header in file '" + filename + "': " + buf;
        fclose(input);
        throw ReadError(errStr);
      }

    if(!big_endian())
      {
	SWAP_32(&(header.magic));
	for(i=0; i<3; i++) SWAP_32(&(header.dim[i]));
	SWAP_32(&(header.numTimesteps));
	SWAP_32(&(header.numVariables));
	for(i=0; i<4; i++) SWAP_32(&(header.min[i]));
	for(i=0; i<4; i++) SWAP_32(&(header.max[i]));
      }

    stat(filename.c_str(), &s);
    
    /* error checking */
    if(header.magic != 0xBAADBEEF) // check for magic number
      {
	fclose(input);
	throw InvalidRawVHeader("Magic number not present");
      }
    if(header.numVariables == 0) // make sure there is more than 1 variable
      {
	fclose(input);
	throw InvalidRawVHeader("numVariables == 0");
      }
    // make sure the file size is bigger than potential header data size
    if(sizeof(header_t)+sizeof(var_record_t)*uint64(header.numVariables)>=uint64(s.st_size))
      {
	fclose(input);
	throw InvalidRawVHeader("File size is smaller than potential header size!");
      }
    // make sure that the file size not greater than the largest possible volume
    //  according to the header.
    if((uint64(sizeof(header_t))+uint64(sizeof(var_record_t))*header.numVariables+
	uint64(header.dim[0]*header.dim[1]*header.dim[2])*rawv_type_sizes[5]*
	header.numTimesteps*header.numVariables)<uint64(s.st_size))
      {
	fclose(input);
	throw InvalidRawVHeader("File size does not match header info");
      }
    // make sure that the file size is at least as big as the smallest possible
    //  volume according to the header
    if((uint64(sizeof(header_t))+uint64(sizeof(var_record_t))*header.numVariables+
	uint64(header.dim[0]*header.dim[1]*header.dim[2])*rawv_type_sizes[1]*
	header.numTimesteps*header.numVariables)>uint64(s.st_size))
      {
	fclose(input);
	throw InvalidRawVHeader("File size does not match header info");
      }

    // now we should be safe to allocate based on the numVariables value in the header
    var_records = new var_record_t[header.numVariables];
    if(var_records == NULL)
      {
	fclose(input);
	throw MemoryAllocationError("Cannot allocate memory for RawV variable records");
      }

    /* read variable records */
    dataBytes = 0;
    for(i = 0; i<header.numVariables; i++)
      {
	// read a single record
	if(fread(&(var_records[i]), sizeof(var_record_t), 1, input) != 1)
	  {
	    geterrstr(errno,buf,256);
	    std::string errStr = "Error reading variable record in file '" + filename + "': " + buf;
	    fclose(input);
	    delete [] var_records;
	    throw ReadError(errStr);
	  }
	
	// check for null byte in variable name
	for(j=0; j<64; j++)
	  if(var_records[i].varName[j] == '\0') break;
	if(j==64)
	  {
	    fclose(input);
	    delete [] var_records;
	    throw InvalidRawVHeader("Non null terminated variable name for variable");
	  }
	
	// make sure that the variable type specified is legal
	if(var_records[i].varType > 5)
	  {
	    fclose(input);
	    delete [] var_records;
	    throw InvalidRawVHeader("Invalid variable type");
	  }

	//count how many bytes this variable uses up so we can check this against the whole file size
	dataBytes += header.dim[0]*header.dim[1]*header.dim[2]*rawv_type_sizes[var_records[i].varType]*header.numTimesteps;
      }
    
    if(sizeof(header_t)+sizeof(var_record_t)*uint64(header.numVariables)+dataBytes != uint64(s.st_size))
      {
	// arand: added type-cast to eliminate warnings 
	SNPRINTF(buf,255,"File size does not match header info: %llu %llu",
		 (long long unsigned int)dataBytes,
		 (long long unsigned int)uint64(s.st_size));

	fclose(input);
	delete [] var_records;
	throw InvalidRawVHeader(buf);
      }

    /*
      At this point we can be sure that this RawV file is correct.  We can now trust header values.
    */
    data._filename = filename;
    data._numVariables = header.numVariables;
    data._numTimesteps = header.numTimesteps;
    data._dimension = Dimension(header.dim);
    data._boundingBox = BoundingBox(header.min[0],header.min[1],header.min[2],
				    header.max[0],header.max[1],header.max[2]);
    data._tmin = header.min[3];
    data._tmax = header.max[3];
    data._voxelTypes.clear();
    data._names.clear();
    for(i=0; i<header.numVariables; i++)
      {
	data._voxelTypes.push_back(rawv_type_conv[var_records[i].varType]);
	data._names.push_back(var_records[i].varName);
      }
   
    /* new volume, so min/max is now unset */
    data._minIsSet.clear();
    data._minIsSet.resize(data._numVariables); for(i=0; i<data._minIsSet.size(); i++) data._minIsSet[i].resize(data._numTimesteps);
    data._min.clear();
    data._min.resize(data._numVariables); for(i=0; i<data._min.size(); i++) data._min[i].resize(data._numTimesteps);
    data._maxIsSet.clear();
    data._maxIsSet.resize(data._numVariables); for(i=0; i<data._maxIsSet.size(); i++) data._maxIsSet[i].resize(data._numTimesteps);
    data._max.clear();
    data._max.resize(data._numVariables); for(i=0; i<data._max.size(); i++) data._max[i].resize(data._numTimesteps);
    
    fclose(input);
    delete [] var_records;
  }

  // -----------------------
  // RawV_IO::readVolumeFile
  // -----------------------
  // Purpose:
  //   Writes to a Volume object after reading from a volume file.
  // ---- Change History ----
  // ??/??/2007 -- Joe R. -- Initial implementation.
  // 11/20/2009 -- Joe R. -- Converted to a VolumeFile_IO class
  void RawV_IO::readVolumeFile(Volume& vol,
			       const std::string& filename, 
			       unsigned int var, unsigned int time,
			       uint64 off_x, uint64 off_y, uint64 off_z,
			       const Dimension& subvoldim) const
  {
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);

    char buf[256];
    VoxelType rawv_type_conv[] = { CVC::UChar, CVC::UChar, CVC::UShort, 
                                   CVC::UInt, CVC::Float, CVC::Double };
    uint64 rawv_type_sizes[] = { 0, 1, 2, 4, 4, 8 };
    
    header_t header;
    var_record_t *var_records;

    FILE *input;
    struct stat s;
    uint64 dataBytes;
    uint64 i,j,k,v;
    
    memset(buf,0,256);

    if(subvoldim.isNull())
      throw IndexOutOfBounds("Specified subvolume dimension is null.");

    if((input = fopen(filename.c_str(),"rb")) == NULL)
      {
	geterrstr(errno,buf,256);
        std::string errStr = "Error opening file '" + filename + "': " + buf;
	throw ReadError(errStr);
      }

    if(fread(&header, sizeof(header_t), 1, input) != 1)
      {
	geterrstr(errno,buf,256);
        std::string errStr = "Error reading header in file '" + filename + "': " + buf;
        fclose(input);
        throw ReadError(errStr);
      }

    if(!big_endian())
      {
	SWAP_32(&(header.magic));
	for(i=0; i<3; i++) SWAP_32(&(header.dim[i]));
	SWAP_32(&(header.numTimesteps));
	SWAP_32(&(header.numVariables));
	for(i=0; i<4; i++) SWAP_32(&(header.min[i]));
	for(i=0; i<4; i++) SWAP_32(&(header.max[i]));
      }

    stat(filename.c_str(), &s);
    
    /* error checking */
    if(header.magic != 0xBAADBEEF) // check for magic number
      {
	fclose(input);
	throw InvalidRawVHeader("Magic number not present");
      }
    if(header.numVariables == 0) // make sure there is more than 1 variable
      {
	fclose(input);
	throw InvalidRawVHeader("numVariables == 0");
      }
    // make sure the file size is bigger than potential header data size
    if(sizeof(header_t)+sizeof(var_record_t)*uint64(header.numVariables)>=uint64(s.st_size))
      {
	fclose(input);
	throw InvalidRawVHeader("File size is smaller than potential header size!");
      }
    // make sure that the file size not greater than the largest possible volume
    //  according to the header.
    if((uint64(sizeof(header_t))+uint64(sizeof(var_record_t))*header.numVariables+
	uint64(header.dim[0]*header.dim[1]*header.dim[2])*rawv_type_sizes[5]*
	header.numTimesteps*header.numVariables)<uint64(s.st_size))
      {
	fclose(input);
	throw InvalidRawVHeader("File size is greater than the largest possible volume according to the header");
      }
    // make sure that the file size is at least as big as the smallest possible
    //  volume according to the header
    if((uint64(sizeof(header_t))+uint64(sizeof(var_record_t))*header.numVariables+
	uint64(header.dim[0]*header.dim[1]*header.dim[2])*rawv_type_sizes[1]*
	header.numTimesteps*header.numVariables)>uint64(s.st_size))
      {
	fclose(input);
	throw InvalidRawVHeader("File size is smaller than the smallest possible volume according to the header");
      }

    // now we should be safe to allocate based on the numVariables value in the header
    var_records = new var_record_t[header.numVariables];
    if(var_records == NULL)
      {
	fclose(input);
	throw MemoryAllocationError("Cannot allocate memory for RawV variable records");
      }

    /* read variable records */
    dataBytes = 0;
    for(i = 0; i<header.numVariables; i++)
      {
	// read a single record
	if(fread(&(var_records[i]), sizeof(var_record_t), 1, input) != 1)
	  {
	    geterrstr(errno,buf,256);
	    std::string errStr = "Error reading variable record in file '" + filename + "': " + buf;
	    fclose(input);
	    delete [] var_records;
	    throw ReadError(errStr);
	  }
	
	// check for null byte in variable name
	for(j=0; j<64; j++)
	  if(var_records[i].varName[j] == '\0') break;
	if(j==64)
	  {
	    fclose(input);
	    delete [] var_records;
	    throw InvalidRawVHeader("Non null terminated variable name for variable");
	  }
	
	// make sure that the variable type specified is legal
	if(var_records[i].varType > 5)
	  {
	    fclose(input);
	    delete [] var_records;
	    throw InvalidRawVHeader("Invalid variable type");
	  }

	//count how many bytes this variable uses up so we can check this against the whole file size
	dataBytes += header.dim[0]*header.dim[1]*header.dim[2]*rawv_type_sizes[var_records[i].varType]*header.numTimesteps;
      }
    
    if(sizeof(header_t)+sizeof(var_record_t)*header.numVariables+dataBytes != uint64(s.st_size))
      {
	// arand: type-cast to eliminate warnings
	SNPRINTF(buf,255,"File size does not match header info: %lld %lld",
		 (long long unsigned int)sizeof(header_t)+sizeof(var_record_t)*header.numVariables+dataBytes,
		 (long long unsigned int)uint64(s.st_size));

	fclose(input);
	delete [] var_records;
	throw InvalidRawVHeader(buf);
      }
   
    // make sure function arguments are correct
    if((off_x + subvoldim[0] - 1 >= header.dim[0]) ||
       (off_y + subvoldim[1] - 1 >= header.dim[1]) ||
       (off_z + subvoldim[2] - 1 >= header.dim[2]))
      {
	fclose(input);
	delete [] var_records;
	throw IndexOutOfBounds("Subvolume specified is outside volume dimensions");
      }

    if(var >= header.numVariables || time >= header.numTimesteps)
      {
	fclose(input);
	delete [] var_records;
	throw IndexOutOfBounds("Requested variable/timestep is larger than the number of variables/timesteps");
      }

    /* 
       At this point we can be sure that this RawV file and function arguments are correct, so we may now modify 'vol'.
    */
    try
      {
	vol.dimension(subvoldim);
	BoundingBox subvolbox;
	subvolbox.setMin(header.min[0]+((header.max[0] - header.min[0])/(header.dim[0] - 1))*off_x,
			 header.min[1]+((header.max[1] - header.min[1])/(header.dim[1] - 1))*off_y,
			 header.min[2]+((header.max[2] - header.min[2])/(header.dim[2] - 1))*off_z);
	subvolbox.setMax(header.min[0]+((header.max[0] - header.min[0])/(header.dim[0] - 1))*(off_x+subvoldim[0]-1),
			 header.min[1]+((header.max[1] - header.min[1])/(header.dim[1] - 1))*(off_y+subvoldim[1]-1),
			 header.min[2]+((header.max[2] - header.min[2])/(header.dim[2] - 1))*(off_z+subvoldim[2]-1));
	vol.boundingBox(subvolbox);
	vol.voxelType(rawv_type_conv[var_records[var].varType]);
	vol.desc(var_records[var].varName);
      }
    catch(MemoryAllocationError& e)
      {
	fclose(input);
	delete [] var_records;
	throw e;
      }

    /*
      Finally read the requested volume data.
    */
    try
      {
	off_t offset = sizeof(header_t)+sizeof(var_record_t)*uint64(header.numVariables);
	off_t file_offx, file_offy, file_offz;
	
	// set offset to the requested variable
	for(v=0; v<var; v++)
	  offset += header.dim[0]*header.dim[1]*header.dim[2]*
	    rawv_type_sizes[var_records[v].varType]*header.numTimesteps;
	
	//set offset to the requested timestep
	offset += header.dim[0]*header.dim[1]*header.dim[2]*
	  rawv_type_sizes[var_records[var].varType]*time;
	
	for(k=off_z; k<=(off_z+subvoldim[2]-1); k++)
	  {
	    file_offz = offset+k*header.dim[0]*header.dim[1]*rawv_type_sizes[var_records[var].varType];
	    for(j=off_y; j<=(off_y+subvoldim[1]-1); j++)
	      {
		file_offy = j*header.dim[0]*rawv_type_sizes[var_records[var].varType];
		file_offx = off_x*rawv_type_sizes[var_records[var].varType];
		//seek and read a scanline at a time
		if(FSEEK(input,file_offx+file_offy+file_offz,SEEK_SET) == -1)
		  {
		    geterrstr(errno,buf,256);
		    std::string errStr = "Error reading volume data in file '" + filename + "': " + buf;
		    throw ReadError(errStr);
		  }
		if(fread(*vol+
			 (k-off_z)*vol.XDim()*vol.YDim()*vol.voxelSize()+
			 (j-off_y)*vol.XDim()*vol.voxelSize(),
			 vol.voxelSize(),vol.XDim(),input) != vol.XDim())
		  {
		    geterrstr(errno,buf,256);
		    std::string errStr = "Error reading volume data in file '" + filename + "': " + buf;
		    throw ReadError(errStr);
		  }
	      }
	  }
      }
    catch(ReadError& e)
      {
	fclose(input);
	delete [] var_records;
	throw e;
      }

    /* swap the volume data if on little endian machine */
    if(!big_endian())
      {
	size_t len = vol.XDim()*vol.YDim()*vol.ZDim();
	switch(vol.voxelType())
	  {
	  case CVC::UShort: for(i=0;i<len;i++) SWAP_16(*vol+i*vol.voxelSize()); break;
	  case CVC::Float:  for(i=0;i<len;i++) SWAP_32(*vol+i*vol.voxelSize()); break;
	  case CVC::Double: for(i=0;i<len;i++) SWAP_64(*vol+i*vol.voxelSize()); break;
	  default: break; /* no swapping needed for unsigned char data, and unsigned int is not defined for rawiv */
	  }
      }

    fclose(input);
    delete [] var_records;
  }

  // -------------------------
  // RawV_IO::createVolumeFile
  // -------------------------
  // Purpose:
  //   Creates an empty volume file to be later filled in by writeVolumeFile
  // ---- Change History ----
  // ??/??/2007 -- Joe R. -- Initial implementation.
  // 11/20/2009 -- Joe R. -- Converted to a VolumeFile_IO class
  void RawV_IO::createVolumeFile(const std::string& filename,
				 const BoundingBox& boundingBox,
				 const Dimension& dimension,
				 const std::vector<VoxelType>& voxelTypes,
				 unsigned int numVariables, unsigned int numTimesteps,
				 double min_time, double max_time) const
  {
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);

    char buf[256];
    unsigned char rawv_inv_type_conv[] = { 1, 2, 3, 4, 5 };

    header_t header;
    var_record_t var_record;

    FILE *output;
    size_t i,j,k,v,t;

    memset(buf,0,256);

    if(boundingBox.isNull())
      throw InvalidBoundingBox("Bounding box must not be null");
    if(dimension.isNull())
      throw InvalidBoundingBox("Dimension must not be null");
    if(voxelTypes.size() != numVariables)
      throw InvalidRawVHeader("Number of variables must match number of supplied voxel types");

    if((output = fopen(filename.c_str(),"wb")) == NULL)
      {
	geterrstr(errno,buf,256);
	std::string errStr = "Error opening file '" + filename + "': " + buf;
	throw WriteError(errStr);
      }

    header.magic = 0xBAADBEEF;
    header.dim[0] = dimension[0];
    header.dim[1] = dimension[1];
    header.dim[2] = dimension[2];
    header.numTimesteps = numTimesteps;
    header.numVariables = numVariables;
    header.min[0] = boundingBox.minx;
    header.min[1] = boundingBox.miny;
    header.min[2] = boundingBox.minz;
    header.min[3] = min_time;
    header.max[0] = boundingBox.maxx;
    header.max[1] = boundingBox.maxy;
    header.max[2] = boundingBox.maxz;
    header.max[3] = max_time;

    if(!big_endian())
      {
	SWAP_32(&(header.magic));
	for(i=0; i<3; i++) SWAP_32(&(header.dim[i]));
	SWAP_32(&(header.numTimesteps));
	SWAP_32(&(header.numVariables));
	for(i=0; i<4; i++) SWAP_32(&(header.min[i]));
	for(i=0; i<4; i++) SWAP_32(&(header.max[i]));
      }

    if(fwrite(&header,sizeof(header),1,output) != 1)
      {
	geterrstr(errno,buf,256);
	std::string errStr = "Error writing header to file '" + filename + "': " + buf;
	fclose(output);
	throw WriteError(errStr);
      }

    // write the variable records
    for(i=0; i<numVariables; i++)
      {
	memset(&var_record,0,sizeof(var_record_t));
	var_record.varType = rawv_inv_type_conv[voxelTypes[i]];
	
	if(fwrite(&var_record,sizeof(var_record),1,output) != 1)
	  {
	    geterrstr(errno,buf,256);
	    std::string errStr = "Error writing header to file '" + filename + "': " + buf;
	    fclose(output);
	    throw WriteError(errStr);
	  }
      }

    //write a scanline at a time
    for(v=0; v<numVariables; v++)
      {
	// each variable may have its own type, so scanline size may differ
	unsigned char * scanline = (unsigned char *)calloc(dimension[0]*VoxelTypeSizes[voxelTypes[v]],1);
	if(scanline == NULL)
	  {
	    fclose(output);
	    throw MemoryAllocationError("Unable to allocate memory for write buffer");
	  }

	for(t=0; t<numTimesteps; t++)
	  for(k=0; k<dimension[2]; k++)
	    for(j=0; j<dimension[1]; j++)
	      {
		if(fwrite(scanline,VoxelTypeSizes[voxelTypes[v]],dimension[0],output) != dimension[0])
		  {
		    geterrstr(errno,buf,256);
		    std::string errStr = "Error writing volume data to file '" + filename + "': " + buf;
		    free(scanline);
		    fclose(output);
		    throw WriteError(errStr);
		  }
	      }
	
	free(scanline);
      }
    
    fclose(output);
  }

  // ------------------------
  // RawV_IO::writeVolumeFile
  // ------------------------
  // Purpose:
  //   Writes the volume contained in wvol to the specified volume file. Should create
  //   a volume file if the filename provided doesn't exist.  Else it will simply
  //   write data to the existing file.  A common user error arises when you try to
  //   write over an existing volume file using this function for unrelated volumes.
  //   If what you desire is to overwrite an existing volume file, first run
  //   createVolumeFile to replace the volume file.
  // ---- Change History ----
  // ??/??/2007 -- Joe R. -- Initial implementation.
  // 11/20/2009 -- Joe R. -- Converted to a VolumeFile_IO class
  void RawV_IO::writeVolumeFile(const Volume& wvol, 
				const std::string& filename,
				unsigned int var, unsigned int time,
				uint64 off_x, uint64 off_y, uint64 off_z) const
  {
    CVC::ThreadInfo ti(BOOST_CURRENT_FUNCTION);

    char buf[256];
    unsigned char rawv_inv_type_conv[] = { 1, 2, 3, 4, 5 };
    VolumeFileInfo volinfo;

    FILE *output;
    size_t i,j,k,v;

    memset(buf,0,256);

    Volume vol(wvol);

    //check if the file exists and we can write the specified subvolume to it
    try
      {
	volinfo.read(filename);
	//if(!(Dimension(off_x+vol.XDim(),off_y+vol.YDim(),off_z+vol.ZDim()) <= volinfo.dimension()))
	if(off_x+vol.XDim() > volinfo.dimension()[0] &&
	   off_y+vol.YDim() > volinfo.dimension()[1] &&
	   off_z+vol.ZDim() > volinfo.dimension()[2])
	  {
	    std::string errStr = "File '" + filename + "' exists but is too small to write volume at specified offset";
	    throw IndexOutOfBounds(errStr);
	  }
	if(var >= volinfo.numVariables())
	  {
	    std::string errStr = "Variable index exceeds number of variables in file '" + filename + "'";
	    throw IndexOutOfBounds(errStr);
	  }
	if(time >= volinfo.numTimesteps())
	  {
	    std::string errStr = "Timestep index exceeds number of timesteps in file '" + filename + "'";
	    throw IndexOutOfBounds(errStr);
	  }
	vol.voxelType(volinfo.voxelTypes(var)); //change the volume's voxel type to match that of the file
      }
    catch(ReadError e)
      {
	//create a blank file since file doesn't exist (or there was an error reading the existing file)
	BoundingBox box(vol.boundingBox());
	box.minx -= off_x * vol.XSpan();
	box.miny -= off_y * vol.YSpan();
	box.minz -= off_z * vol.ZSpan();
	Dimension dim(vol.dimension());
	dim[0] += off_x;
	dim[1] += off_y;
	dim[2] += off_z;
	createVolumeFile(filename,box,dim,std::vector<VoxelType>(1,vol.voxelType()),1,1,0.0,0.0);
	volinfo.read(filename);

	if(var >= volinfo.numVariables())
	  {
	    std::string errStr = "Variable index exceeds number of variables in file '" + filename + "'";
	    throw IndexOutOfBounds(errStr);
	  }
	if(time >= volinfo.numTimesteps())
	  {
	    std::string errStr = "Timestep index exceeds number of timesteps in file '" + filename + "'";
	    throw IndexOutOfBounds(errStr);
	  }
      }

    if((output = fopen(filename.c_str(),"r+b")) == NULL)
      {
	geterrstr(errno,buf,256);
	std::string errStr = "Error opening file '" + filename + "': " + buf;
	throw WriteError(errStr);
      }

    //write the volume record for this volume
    {
      var_record_t var_record;

      if(FSEEK(output,sizeof(header_t)+sizeof(var_record_t)*var,SEEK_SET) == -1)
	{
	  geterrstr(errno,buf,256);
	  std::string errStr = "Error seeking in file '" + filename + "': " + buf;
	  fclose(output);
	  throw WriteError(errStr);
	}

      memset(&var_record,0,sizeof(var_record_t));
      var_record.varType = rawv_inv_type_conv[vol.voxelType()];
      strncpy(var_record.varName,vol.desc().c_str(),64);
      var_record.varName[63] = '\0';

      if(fwrite(&var_record,sizeof(var_record),1,output) != 1)
	{
	  geterrstr(errno,buf,256);
	  std::string errStr = "Error writing header to file '" + filename + "': " + buf;
	  fclose(output);
	  throw WriteError(errStr);
	}
    }

    unsigned char * scanline = (unsigned char *)malloc(vol.XDim()*vol.voxelSize());
    if(scanline == NULL)
      {
	fclose(output);
	throw MemoryAllocationError("Unable to allocate memory for write buffer");
      }

    /*
      write the volume data.
    */
    off_t offset = sizeof(header_t)+sizeof(var_record_t)*volinfo.numVariables();
    off_t file_offx, file_offy, file_offz;
    
    // set offset to the requested variable
    for(v=0; v<var; v++)
      offset += volinfo.XDim()*volinfo.YDim()*volinfo.ZDim()*
	volinfo.voxelSizes(v)*volinfo.numTimesteps();
    
    //set offset to the requested timestep
    offset += volinfo.XDim()*volinfo.YDim()*volinfo.ZDim()*
      volinfo.voxelSizes(var)*time;
    
    for(k=off_z; k<=(off_z+vol.ZDim()-1); k++)
      {
	file_offz = offset+k*volinfo.XDim()*volinfo.YDim()*volinfo.voxelSizes(var);
	for(j=off_y; j<=(off_y+vol.YDim()-1); j++)
	  {
	    file_offy = j*volinfo.XDim()*volinfo.voxelSizes(var);
	    file_offx = off_x*volinfo.voxelSizes(var);
	    //seek and read a scanline at a time
	    if(FSEEK(output,file_offx+file_offy+file_offz,SEEK_SET) == -1)
	      {
		geterrstr(errno,buf,256);
		std::string errStr = "Error seeking in file '" + filename + "': " + buf;
		fclose(output);
		throw WriteError(errStr);
	      }

	    memcpy(scanline,*vol+
		   ((k-off_z)*vol.XDim()*vol.YDim()*vol.voxelSize())+
		   ((j-off_y)*vol.XDim()*vol.voxelSize()),
		   vol.XDim()*vol.voxelSize());

	    /* swap the volume data if on little endian machine */
	    if(!big_endian())
	      {
		size_t len = vol.XDim();
		switch(vol.voxelType())
		  {
		  case CVC::UShort: for(i=0;i<len;i++) SWAP_16(scanline+i*vol.voxelSize()); break;
		  case CVC::Float:  for(i=0;i<len;i++) SWAP_32(scanline+i*vol.voxelSize()); break;
		  case CVC::Double: for(i=0;i<len;i++) SWAP_64(scanline+i*vol.voxelSize()); break;
		  default: break; /* no swapping needed for unsigned char data, and unsigned int is not defined for rawiv */
		  }
	      }

	    if(fwrite(scanline,vol.voxelSize(),vol.XDim(),output) != vol.XDim())
	      {
		geterrstr(errno,buf,256);
		std::string errStr = "Error writing volume data to file '" + filename + "': " + buf;
		free(scanline);
		fclose(output);
		throw WriteError(errStr);
	      }
	  }
      }
  
    free(scanline);
    fclose(output);
  }
};

/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Volume Rover.

  Volume Rover is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  Volume Rover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

// MrcFileImpl.h: interface for the MrcFileImpl class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_MRCFILEIMPL_H__4D6E7F11_7498_4CB1_8D75_F2840E54EDD4__INCLUDED_)
#define AFX_MRCFILEIMPL_H__4D6E7F11_7498_4CB1_8D75_F2840E54EDD4__INCLUDED_

#include <VolumeFileTypes/BasicVolumeFileImpl.h>

#ifdef LARGEFILE_KLUDGE
#include <VolumeFileTypes/pfile.h>
#else
#include <qfile.h>
#endif

///\struct OLDMrcHeader
///\brief The old-style header of an MRC file.
typedef struct {
//! # of columns ( fastest changing in the map    
	int    nx;            
//! # of rows                                     
	int    ny;           
//! # of sections (slowest changing in the map    
	int    nz;
	
//! data type
//! 0 = image data in bytes
//! 1 = image data in short integer
//! 2 = image data in floats
//! 3 = complex data in complex short integers
//! 4 = complex data in complex reals          
	int    mode;
	
//! number of first column in map (default = 0)   
	int    nxstart;
//! number of first row in map (default = 0)      
	int    nystart;
//! number of first ssection in map (default = 0) 
	int    nzstart;
	
//! number of intervals along X                   
	int    mx;
//! number of intervals along Y                   
	int    my;
//! number of intervals along Z                   
	int    mz;
	
//! cell dimensions in X (angstrom)               
	float  xlength;
//! cell dimensions in Y (angstrom)               
	float  ylength;
//! cell dimensions in Z (angstrom)               
	float  zlength;
	
//! cell angles between Y and Z                   
	float  alpha;
//! cell angles between X and Z                   
	float  beta;
//! cell angles between X and Y                   
	float  gamma;
	
//! number of axis corresponding to columns (X)   
	int    mapc;
//! number of axis corresponding to rows (Y)      
	int    mapr;
//! number of axis corresponding to sections (Z)  
	int    maps;
	
//! minimum density value                         
	float  amin;
//! maximum density value                         
	float  amax;
//! mean density value                            
	float  amean;
	
//! space group number (0 for images)             
	int    ispg;
//! # of bytes for symmetry operators             
	int    nsymbt;
	
//! user defined storage space                    
	int    extra[29];
	
//! X phase origin                                
	float  xorigin;
//! Y phase origin                                
	float  yorigin;
	
//! # of labels being used in the MRC header      
	int    nlabl;
	
//! actual text labels                            
	char   label[10][80];
	
} OLDMrcHeader;

///\struct MrcHeader
///\brief The header of an MRC file.
typedef struct {
	
//! # of columns ( fastest changing in the map    
	int    nx;
//! # of rows                                     
	int    ny;
//! # of sections (slowest changing in the map    
	int    nz;
	
//! data type
//! 0 = image data in bytes
//! 1 = image data in short integer
//! 2 = image data in floats
//! 3 = complex data in complex short integers
//! 4 = complex data in complex reals          
	int    mode;
	
//! number of first column in map (default = 0)   
	int    nxstart;
//! number of first row in map (default = 0)      
	int    nystart;
//! number of first ssection in map (default = 0) 
	int    nzstart;
	
//! number of intervals along X                   
	int    mx;
//! number of intervals along Y                   
	int    my;
//! number of intervals along Z                   
	int    mz;
	
//! cell dimensions in X (angstrom)               
	float  xlength;
//! cell dimensions in Y (angstrom)               
	float  ylength;
//! cell dimensions in Z (angstrom)               
	float  zlength;
	
//! cell angles between Y and Z                   
	float  alpha;
//! cell angles between X and Z                   
	float  beta;
//! cell angles between X and Y                   
	float  gamma;
	
//! number of axis corresponding to columns (X)   
	int    mapc;
//! number of axis corresponding to rows (Y)      
	int    mapr;
//! number of axis corresponding to sections (Z)  
	int    maps;
	
//! minimum density value                         
	float  amin;
//! maximum density value                         
	float  amax;
//! mean density value                            
	float  amean;
	
//! space group number (0 for images)             
	int    ispg;
//! # of bytes for symmetry operators             
	int    nsymbt;
	
//! user defined storage space                    
	int    extra[25];
	
//! X phase origin                                
	float  xorigin;
//! Y phase origin                                
	float  yorigin;
//! Z phase origin
	float  zorigin;

//! character string 'MAP '
	char   map[4];

//! machine stamp
	int    machst;

//! rms deviation of map from mean density
	float  rms;
	
//! # of labels being used in the MRC header      
	int    nlabl;
	
//! actual text labels                            
	char   label[10][80];
	
} MrcHeader;

typedef struct
{
  float aTilt;
  float bTilt;
  float xStage;
  float yStage;
  float zStage;
  float xShift;
  float yShift;
  float defocus;
  float expTime;
  float meanInt;
  float tiltAxis;
  float pixelSize;
  float imageMag;
  char filler[76];
} ExtendedMrcHeader;

///\class MrcFileImpl MrcFileImpl.h
///\brief A BasicVolumeFileImpl instance that reads and writes MRC files.
///\author Anthony Thane
///\author John Wiggins
class MrcFileImpl : public BasicVolumeFileImpl  
{
public:
	MrcFileImpl();
	virtual ~MrcFileImpl();

	virtual bool checkType(const QString& fileName);
	virtual bool attachToFile(const QString& fileName, Mode mode = Read);

	virtual QString getExtension() { return "mrc"; };
	virtual QString getFilter() { return "MRC files (*.mrc)"; };

	virtual void setVariableType(unsigned int variable, Type type);
				
	virtual bool readCharData(unsigned char* buffer, unsigned int numSamples, unsigned int variable);
	virtual bool readShortData(unsigned short* buffer, unsigned int numSamples, unsigned int variable);
	virtual bool readLongData(unsigned int* buffer, unsigned int numSamples, unsigned int variable);
	virtual bool readFloatData(float* buffer, unsigned int numSamples, unsigned int variable);
	virtual bool readDoubleData(double* buffer, unsigned int numSamples, unsigned int variable);

	virtual bool writeHeader();

	virtual bool writeCharData(unsigned char* buffer, unsigned int numSamples, unsigned int variable);
	virtual bool writeShortData(unsigned short* buffer, unsigned int numSamples, unsigned int variable);
	virtual bool writeLongData(unsigned int* buffer, unsigned int numSamples, unsigned int variable);
	virtual bool writeFloatData(float* buffer, unsigned int numSamples, unsigned int variable);
	virtual bool writeDoubleData(double* buffer, unsigned int numSamples, unsigned int variable);

	virtual VolumeFile* getNewVolumeFileLoader() const;
protected:
	virtual bool protectedSetPosition(unsigned int variable, unsigned int timeStep, Q_ULLONG offset);

	bool readHeader(Q_ULLONG fileSize);
	bool checkHeader(MrcHeader& header, Q_ULLONG fileSize);
	bool interpretNewHeader(MrcHeader& header, Q_ULLONG fileSize);
	bool interpretOldHeader(MrcHeader& header, Q_ULLONG fileSize);
	void swapHeader(MrcHeader& header);
	void fillHeader(MrcHeader& header);
	void close();
	unsigned int headerSize(unsigned int numVariables) const;

#ifdef LARGEFILE_KLUDGE
	PFile m_File;
#else
	QFile m_File;
#endif
	Mode m_OpenMode;
	bool m_Attached;
	bool m_MustSwap;

};

#endif // !defined(AFX_MRCFILEIMPL_H__4D6E7F11_7498_4CB1_8D75_F2840E54EDD4__INCLUDED_)

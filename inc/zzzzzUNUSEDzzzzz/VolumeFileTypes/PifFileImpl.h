/*
  Copyright 2002-2005 The University of Texas at Austin
  
	Authors: John Wiggins <prok@ices.utexas.edu>
           Anthony Thane <thanea@ices.utexas.edu>
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
  along with Volume Rover; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

// PifFileImpl.h: interface for the PifFileImpl class.
//
//////////////////////////////////////////////////////////////////////

#ifndef PIF_FILE_IMPL_H
#define PIF_FILE_IMPL_H

#include <VolumeFileTypes/BasicVolumeFileImpl.h>

#ifdef LARGEFILE_KLUDGE
#include <VolumeFileTypes/pfile.h>
#else
#include <qfile.h>
#endif

///\struct PifFileHeader
///\brief This is the header for a PIF file.
struct PifFileHeader{
	char	file_id[8];				//	8 bytes
	char	RealScaleFactor[16];	//	24
	int		numImages;				//	28 - number of maps in this file
	int		endianNess;				//	32
			// 0 - little endian
			// 1 - big endian
	char	genProgram[32];			//	64 - the program that generated the file
	int		htype;					//	68
			// 1 - all projections have the same number of pixels
			//		and the same depth (number of bits per pixel).
			// 0 - otherwise
	int		nx;						//	72 - X dim in voxels
	int		ny;						//	76 - Y dim in voxels
	int		nz;						//	80 - Z dim in voxels
			// Should the previous fields be ignored?
			// The "same" data is repeated in the image header.
			// My guess is that htype tells us whether or not we can ignore them.
	int		mode;					//	84
			//	Em data type
			//		0 = byte	-	Zeiss scans
			//		1 = short
			//		2 = float or int (WTF? apples or oranges?)
			//		3 = complex short
			//		4 = complex float or int
			//		5 = structure factors
			//		6 = boxed data
			//		7 = short or 2-byte float
			//		8 = complex short or 2-byte float
			//		9 = float
			//	   10 = complex float
			//	   20 = MAP short or 2-byte float
			//	   21 = MAP float or int
			//	   22 = MAP float or int PFTS rot*4 dimension
			//	   31 = structure factors amp/phase float or int
			//	   32 = structure factors Apart/Bpart float or int
			//	   88 = accumulated TIF's in short
			//	   97 = depthcued, 1536 byte colormap follows the file header
	char	futureUse1[428];		//	512
};

///\struct PifImageHeader
///\brief This is the header for a PIF image. A PIF file may contain more than
///	one image.
struct PifImageHeader{
	int		nx;						//	4
	int		ny;						//	8
	int		nz;						//	12
	int		mode;					//	16
			//	Em data type
			//		0 = byte	-	Zeiss scans
			//		1 = short
			//		2 = float or int
			//		3 = complex short
			//		4 = complex float or int
			//		5 = structure factors
			//		6 = boxed data
			//		7 = short or 2-byte float
			//		8 = complex short or 2-byte float
			//		9 = float
			//	   10 = complex float
			//	   20 = MAP short or 2-byte float
			//	   21 = MAP float or int
			//	   22 = MAP float or int PFTS rot*4 dimension
			//	   31 = structure factors amp/phase float or int
			//	   32 = structure factors Apart/Bpart float or int
			//	   88 = accumulated TIF's in short
			//	   97 = depthcued, 1536 byte colormap follows the file header
	int		bkgnd;					//	20 - background value
	int		packRadius;				//	24 - radius of boxed image
	int		nxstart;				//	28 - # of first column in map
	int		nystart;				//	32 - # of first row in map
	int		nzstart;				//	36 - # of first section in map
	int		mx;						//	40 - # of intervals along X
	int		my;						//	44 - # of intervals along Y
	int		mz;						//	48 - # of intervals along Z
	float	xlength;				//	52 - cell dimensions in angstroms
	float	ylength;				//	56 - "
	float	zlength;				//	60 - "
	int		alpha;					//	64 - cell angles in degrees
	int		beta;					//	68 - "
	int		gamma;					//	72 - "
	int		mapc;					//	76 - which axis is columns? (1,2,3 = x,y,z)
	int		mapr;					//	80 - which axis is rows? (1,2,3 = x,y,z)
	int		maps;					//	84 - which axis is sections? (1,2,3, = x,y,z)
	int		min;					//	88 - min density value
	int		max;					//	92 - max density value
	int		mean;					//	96 - mean density value
	int		stdDev;					//	100 - standard deviation of gray levels
	int		ispg;					//	104 - space group number
	int		nsymbt;					//	108 - # of bytes for symetry ops
	int		xorigin;				//	112 - x origin
	int		yorigin;				//	116 - y origin
	char	title[80];				//	196 - user defined description
	char	timeStamp[32];			//	228 - date/time data last modified
	char	microGraphDesignation[16];	//244 - unique micrograph number
	char	scanNumber[8];			//	252 - scan number of micrograph
	int		aoverb;					//	256
	int		map_abang;				//	260
	int		dela;					//	264
	int		delb;					//	268
	int		delc;					//	272
	int		t_matrix[6];			//	296
	int		dthe;					//	300
	int		dphi_90;				//	304
	int		symmetry;				//	308
	int		binFactor;				//	312 - image compression factor
	int		a_star;					//	316 - emsf3dbt/emmap3dbt stuff
	int		b_star;					//	320
	int		c_star;					//	324
	int		alp_star;				//	328
	int		bet_star;				//	332
	int		gam_star;				//	336
	int		pixelSize;				//	340 - from em3dr
	char	futureUse1[172];		//	512
};

///\class PifFileImpl PifFileImpl.h
///\brief A BasicVolumeFileImpl instance that reads and writes PIF files.
///\author John Wiggins
class PifFileImpl : public BasicVolumeFileImpl  
{
public:
	PifFileImpl();
	virtual ~PifFileImpl();

	virtual bool checkType(const QString& fileName);
	virtual bool attachToFile(const QString& fileName, Mode mode = Read);

	virtual QString getExtension() { return "pif"; };
	virtual QString getFilter() { return "PIF files (*.pif)"; };

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
	bool checkHeader(PifFileHeader& header, Q_ULLONG fileSize);
	bool interpretHeaders(PifFileHeader& fheader, PifImageHeader& iheader, Q_ULLONG fileSize);
	void swapFileHeader(PifFileHeader& header);
	void swapImageHeader(PifImageHeader& header);
	void fillHeaders(PifFileHeader& fheader, PifImageHeader& iheader);
	void close();
	unsigned int headerSize(unsigned int numVariables) const;

#ifdef LARGEFILE_KLUDGE
	PFile m_File;
#else
	QFile m_File;
#endif
	Mode m_OpenMode;
	float m_ScaleFactor;
	bool m_Attached;
	bool m_MustSwap;

	// for scaling short -> float or int -> float
	bool m_MustScale;
	unsigned char m_BytesPerVoxel;

};

#endif


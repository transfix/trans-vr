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

// RawVFileImpl.h: interface for the RawVFileImpl class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_RAWVFILEIMPL_H__13740B05_48CA_48E2_8A69_841EF8F609C8__INCLUDED_)
#define AFX_RAWVFILEIMPL_H__13740B05_48CA_48E2_8A69_841EF8F609C8__INCLUDED_

#include <VolumeFileTypes/BasicVolumeFileImpl.h>

#ifdef LARGEFILE_KLUDGE
#include <VolumeFileTypes/pfile.h>
#else
#include <qfile.h>
#endif

///\class RawVFileImpl RawVFileImpl.h
///\brief A BasicVolumeFileImpl instance that reads and writes RawV files.
///\author Anthony Thane
///\author John Wiggins
class RawVFileImpl : public BasicVolumeFileImpl  
{
public:
	RawVFileImpl();
	virtual ~RawVFileImpl();

	virtual bool checkType(const QString& fileName);
	virtual bool attachToFile(const QString& fileName, Mode mode = Read);

	virtual QString getExtension() { return "rawv"; };
	virtual QString getFilter() { return "RawV files (*.rawv)"; };
	
	virtual void setNumVariables(unsigned int number);

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
	void close();
	unsigned int headerSize(unsigned int numVariables) const;

#ifdef LARGEFILE_KLUDGE
	PFile m_File;
#else
	QFile m_File;
#endif
	Mode m_OpenMode;
	bool m_Attached;
	Q_ULLONG* m_VariableStartingLocations;

};

#endif // !defined(AFX_RAWVFILEIMPL_H__13740B05_48CA_48E2_8A69_841EF8F609C8__INCLUDED_)

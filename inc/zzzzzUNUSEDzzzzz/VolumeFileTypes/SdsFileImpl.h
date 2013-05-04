// SdsFileImpl.h: interface for the SdsFileImpl class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_SDSFILEIMPL_H__D0CA8A5B_28A4_448F_9E81_E76613B3D6D4__INCLUDED_)
#define AFX_SDSFILEIMPL_H__D0CA8A5B_28A4_448F_9E81_E76613B3D6D4__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <VolumeFileTypes/BasicVolumeFileImpl.h>
#include <qfile.h>

///\class SdsFileImpl SdsFileImpl.h
///\brief A BasicVolumeFileImpl instance that reads and writes SDS files.
///\deprecated This class is no longer being used.
///\author Anthony Thane
class SdsFileImpl : public BasicVolumeFileImpl  
{
public:
	SdsFileImpl();
	virtual ~SdsFileImpl();

	virtual bool checkType(const QString& fileName);
	virtual bool attachToFile(const QString& fileName);

	virtual QString getExtension() { return "sds"; };
	virtual QString getFilter() { return "SDS files (*.sds)"; };

	virtual bool readCharData(unsigned char* buffer, unsigned int numSamples, unsigned int variable);
	virtual bool readShortData(unsigned short* buffer, unsigned int numSamples, unsigned int variable);
	virtual bool readLongData(unsigned int* buffer, unsigned int numSamples, unsigned int variable);
	virtual bool readFloatData(float* buffer, unsigned int numSamples, unsigned int variable);
	virtual bool readDoubleData(double* buffer, unsigned int numSamples, unsigned int variable);

	virtual VolumeFile* getNewVolumeFileLoader() const;
protected:
	virtual bool protectedSetPosition(unsigned int variable, unsigned int timeStep, Q_ULLONG offset);

	int readHeader(QFile& file);
	void close();
	unsigned int headerSize(unsigned int numVariables) const;

	QFile m_File;
	bool m_Attached;
};


//	SAIL DATA SET
#define SAIL_DATA_SET_VERSION 2

/*	version 1 obsolete		*/
/***********************************************************************/
typedef enum
{	SDS_READ,		// read header and data (allocation by reader)
	SDS_READ_HEADER,	// read header (allocation by reader)
	SDS_READ_DATA		// read data (allocation by calling program)
} sdsReadFunction;
/***********************************************************************/
typedef enum
{	sdsGeneric,
	sdsLattice,
} sdsDataFormat;

/***********************************************************************/
typedef enum
{	sdsDataType_byte8,
	sdsDataType_short16,
    	sdsDataType_int32,
    	sdsDataType_float32,
    	sdsDataType_double64,
   	sdsDataType_ascii
} sdsDataType;

long	sdsDataSize[7]={1,2,4,4,8,8,1};
/***********************************************************************/
typedef enum
{    	sdsCoordType_uniform,
    	sdsCoordType_perimeter,
    	sdsCoordType_curvilinear
} sdsCoordType;
/***********************************************************************/
typedef struct sdsDataHeader
{	float	 	pi_endian;
	int		typeDef;

	union
	{	struct
		{	int	nDesc;
			int	dataSetSize;
		}Generic;

		struct
		{	int	nDesc;
			int	dataSetSize;
			int	nDim;
			int	dataType;	
			int	coordType;
			int	nDataVar;
			int	nCoordVar;
			int	nData;
			int	nCoord;	
		}sdsLattice;
	}df;
}sdsDataHeader;
/***********************************************************************/
typedef struct sdsDataPointers
{	union
	{	struct
		{	char	*desc;
			int	*data;	
		}Generic;

		struct
		{	char	*desc;
			int	*dims;
			void	*data;
			float	*coord;			
		}sdsLattice;
	}df;
}sdsDataPointers;
/***********************************************************************/
typedef struct sdsFile
{	char		*fileInfo;		// null terminated
	char		*userInfo;		// null terminated
	char		*creationInfo;		// null terminated
	int		nDataSets;
	sdsDataHeader	*dh;
	sdsDataPointers	*ptr;
}sdsFile;
/************************************************************************/
/*  example        1234567...                                           */
//fileInfo     -->|SAIL DATA SET version=1 infoMaxLineSize=512 nDataSets=1 headerSize=xxx addressing=32
//userInfo     -->|optional user description ....					
//creationInfo -->|automated creation information .....
//
// sdsLattice dataset
// followed by a data header		(sizeof(sdsDataHeader))
// followed by description		(sizeof(char)*strlen(description))
// followed by a vector of dims data 	(sizeof(int)*nDim)
// followed by dataVals 		(sizeof(dataType)*nDataVar*dims[0]*...*dims[nDims-1])
// followed by coordVals
//	sdsCoordType_uniform 		(sizeof(float)*nDim*2)
//	sdsCoordType_perimeter		(sizeof(float)*dims[0]+...+dims[nDims-1])
//	sdsCoordType_curvilinear	(sizeof(float)*dims[0]*...*dims[nDims-1]*nCoordVar)
//
// followed by next dataset ...
/************************************************************************/


#endif // !defined(AFX_SDSFILEIMPL_H__D0CA8A5B_28A4_448F_9E81_E76613B3D6D4__INCLUDED_)

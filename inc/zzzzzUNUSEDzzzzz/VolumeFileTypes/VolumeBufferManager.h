/******************************************************************************
				Copyright   

This code is developed within the Computational Visualization Center at The 
University of Texas at Austin.

This code has been made available to you under the auspices of a Lesser General 
Public License (LGPL) (http://www.ices.utexas.edu/cvc/software/license.html) 
and terms that you have agreed to.

Upon accepting the LGPL, we request you agree to acknowledge the use of use of 
the code that results in any published work, including scientific papers, 
films, and videotapes by citing the following references:

C. Bajaj, Z. Yu, M. Auer
Volumetric Feature Extraction and Visualization of Tomographic Molecular Imaging
Journal of Structural Biology, Volume 144, Issues 1-2, October 2003, Pages 
132-143.

If you desire to use this code for a profit venture, or if you do not wish to 
accept LGPL, but desire usage of this code, please contact Chandrajit Bajaj 
(bajaj@ices.utexas.edu) at the Computational Visualization Center at The 
University of Texas at Austin for a different license.
******************************************************************************/

// VolumeBufferManager.h: interface for the VolumeBufferManager class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_VOLUMEBUFFERMANAGER_H__D6F3680E_C8F9_49E2_95A9_A1BAC0426CE4__INCLUDED_)
#define AFX_VOLUMEBUFFERMANAGER_H__D6F3680E_C8F9_49E2_95A9_A1BAC0426CE4__INCLUDED_

class VolumeBuffer;
class SourceManager;
#include <VolumeFileTypes/DownLoadManager.h>

///\class VolumeBufferManager VolumeBufferManager.h
///\author Anthony Thane
///\brief The VolumeBufferManager manages a collection of VolumeBuffer objects (one for each variable).
/// It uses both SourceManager and DownLoadManager to do its dirty work. In NewVolumeMainWindow it 
/// mainly provides abstraction for DownLoadManager.
class VolumeBufferManager  
{
public:
	VolumeBufferManager();
	virtual ~VolumeBufferManager();

///\fn unsigned int getNumberOfVariables() const
///\brief Returns the number of variables in the current dataset
///\return The number of variables
	unsigned int getNumberOfVariables() const;

///\fn VolumeBuffer* getVolumeBuffer(unsigned int var)
///\brief Returns a pointer to a VolumeBuffer containing volume data
///\param var A variable index
///\return A pointer to a VolumeBuffer object
	VolumeBuffer* getVolumeBuffer(unsigned int var);
///\fn VolumeBuffer* getGradientBuffer(unsigned int var)
///\brief Returns a pointer to a VolumeBuffer containing gradient data
///\param var A variable index
///\return A pointer to a VolumeBuffer object
	VolumeBuffer* getGradientBuffer(unsigned int var);

///\fn bool setRequestRegion(double minX, double minY, double minZ, double maxX, double maxY, double maxZ, unsigned int time)
///\brief This function sets the region of the function (volume) that data is extracted from. Spatial 
/// coordinates are specified. The returned volumes can be from any level of the 3D mip map. (ie- they
/// will be at some arbitrary resolution).
///\param minX The minimum X coordinate
///\param minY The minimum Y coordinate
///\param minZ The minimum Z coordinate
///\param maxX The maximum X coordinate
///\param maxY The maximum Y coordinate
///\param maxZ The maximum Z coordinate
///\param time The time step
///\return A bool indicating success or failure
	bool setRequestRegion(double minX, double minY, double minZ, double maxX, double maxY, double maxZ, unsigned int time);
///\fn bool setMaxResolution(unsigned int dimX, unsigned int dimY, unsigned int dimZ)
///\brief This function sets the maximum resolution of the VolumeBuffer objects used by the class.
///\param dimX The dimension in X
///\param dimY The dimension in Y
///\param dimZ The dimension in Z
///\return A bool indicating success or failure
	bool setMaxResolution(unsigned int dimX, unsigned int dimY, unsigned int dimZ);
///\fn bool setSourceManager(SourceManager* sourceManager)
///\brief This function assigns a SourceManager object to the class
///\param sourceManager A pointer to a SourceManager
///\return A bool indicating success or failure
	bool setSourceManager(SourceManager* sourceManager);
///\fn SourceManager* getSourceManager()
///\brief Returns a pointer to the class' SourceManager instance
///\return A pointer to a SourceManager
	SourceManager* getSourceManager();
///\fn DownLoadManager* getDownLoadManager()
///\brief Returns a pointer to the class' DownLoadManager instance
///\return A pointer to a DownLoadManager
	DownLoadManager* getDownLoadManager();

///\fn void markBufferAsInvalid(unsigned int var)
///\brief This function invalidates buffer do that they will be re-fetched from their source.
///\param var The variable whose buffer should be invalidated
	void markBufferAsInvalid(unsigned int var);

protected:
	bool setNumberOfVariables(unsigned int numVariables);
	bool isBufferUpToDate(unsigned int var) const;
	bool isGradientBufferUpToDate(unsigned int var) const;
	bool reallocate(unsigned int numVariables);
	void invalidateBuffers();

	bool downLoadVariable(unsigned int Variable);
	bool downLoadGradient(unsigned int Variable);

	VolumeBuffer** m_VolumeBuffers;
	VolumeBuffer* m_GradientBuffer;
	bool* m_VolumeUpToDate;
	unsigned int m_NumVariables;
	unsigned int m_GradientVariable;

	unsigned int m_NumAllocated;

	unsigned int m_DimX, m_DimY, m_DimZ;
	double m_MinX, m_MinY, m_MinZ;
	double m_MaxX, m_MaxY, m_MaxZ;
	unsigned int m_Time;

	DownLoadManager m_DownLoadManager;
	SourceManager* m_SourceManager;

	bool m_GradientUpToDate;

};

#endif // !defined(AFX_VOLUMEBUFFERMANAGER_H__D6F3680E_C8F9_49E2_95A9_A1BAC0426CE4__INCLUDED_)

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

// VolumeBufferManager.cpp: implementation of the VolumeBufferManager class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeFileTypes/VolumeBufferManager.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

VolumeBufferManager::VolumeBufferManager()
{
	m_NumAllocated = 0;
	m_NumVariables = 0;
	m_VolumeBuffers = 0;
	m_GradientBuffer = new VolumeBuffer;
	m_VolumeUpToDate = 0;
	m_GradientVariable = 0;
	m_GradientUpToDate = false;
	m_SourceManager = 0;
	m_DimX = m_DimY = m_DimZ = 0;
	m_MinX = m_MinY = m_MinZ = 0.0;
	m_MaxX = m_MaxY = m_MaxZ = 0.0;
	setMaxResolution(256, 256, 256);
	setRequestRegion(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0);
	
}

VolumeBufferManager::~VolumeBufferManager()
{
	unsigned int c;
	for (c=0; c<m_NumAllocated; c++) {
		delete m_VolumeBuffers[c];
	}
	delete [] m_VolumeBuffers;
	delete m_GradientBuffer;
	delete [] m_VolumeUpToDate;
}

unsigned int VolumeBufferManager::getNumberOfVariables() const
{
	return m_NumVariables;
}

VolumeBuffer* VolumeBufferManager::getVolumeBuffer(unsigned int var)
{
	if (var<m_NumVariables) {
		if (isBufferUpToDate(var) || downLoadVariable(var)) {
			return m_VolumeBuffers[var];
		}
		else {
			return 0;
		}
	}
	else {
		return 0;
	}
}

VolumeBuffer* VolumeBufferManager::getGradientBuffer(unsigned int var)
{
	if (m_GradientBuffer) {
		if (isGradientBufferUpToDate(var) || downLoadGradient(var)) {
			return m_GradientBuffer;
		}
		else {
			return 0;
		}
	}
	else {
		return 0;
	}
}

bool VolumeBufferManager::setRequestRegion(double minX, double minY, double minZ, double maxX, double maxY, double maxZ, unsigned int time)
{
	if (m_MinX!=minX ||
		m_MinY!=minY ||
		m_MinZ!=minZ ||
		m_MaxX!=maxX ||
		m_MaxY!=maxY ||
		m_MaxZ!=maxZ ||
		m_Time!=time) {
		invalidateBuffers();
	}

	m_MinX = minX;
	m_MinY = minY;
	m_MinZ = minZ;
	m_MaxX = maxX;
	m_MaxY = maxY;
	m_MaxZ = maxZ;
	m_Time = time;
	//invalidateBuffers();
	return true;
}

bool VolumeBufferManager::setMaxResolution(unsigned int dimX, unsigned int dimY, unsigned int dimZ)
{
	m_DimX = dimX;
	m_DimY = dimY;
	m_DimZ = dimZ;
	m_DownLoadManager.allocateBuffer(dimX, dimY, dimZ);
	invalidateBuffers();
	return true;
}

bool VolumeBufferManager::setSourceManager(SourceManager* sourceManager)
{
	if (sourceManager) {
		invalidateBuffers();
		m_SourceManager = sourceManager;

		if (!setNumberOfVariables(m_SourceManager->getSource()->getNumVars())) {
			m_SourceManager = 0;
			return false;
		}
		else {
			return true;
		}
	}
	else {
		sourceManager = 0;
		m_NumVariables = 0;
		return true;
	}
}

SourceManager* VolumeBufferManager::getSourceManager()
{
	return m_SourceManager;
}

DownLoadManager* VolumeBufferManager::getDownLoadManager()
{
	return &m_DownLoadManager;
}

bool VolumeBufferManager::setNumberOfVariables(unsigned int numVariables)
{
	if ((numVariables>m_NumAllocated) && reallocate(numVariables)) {
		m_NumVariables = numVariables;
		return true;
	}
	else {
		m_NumVariables = numVariables;
		return true;
	}
}

bool VolumeBufferManager::isBufferUpToDate(unsigned int var) const
{
	return (var<m_NumVariables && m_VolumeUpToDate[var]);
}

bool VolumeBufferManager::isGradientBufferUpToDate(unsigned int var) const
{
	return (var == m_GradientVariable && m_GradientUpToDate);
}

bool VolumeBufferManager::reallocate(unsigned int numVariables)
{
	// this function should only be called to grow the array
	if (numVariables<=m_NumAllocated) return false;

	VolumeBuffer** temp = new VolumeBuffer*[numVariables];
	if (!temp) {
		return false;
	}
	delete [] m_VolumeUpToDate;
	m_VolumeUpToDate = new bool[numVariables];
	if (!temp) {
		return false;
	}

	unsigned int c;
	unsigned int numToCopy = (numVariables<m_NumAllocated?numVariables:m_NumAllocated);

	// copy existing ones
	for (c=0; c<numToCopy; c++) {
		temp[c] = m_VolumeBuffers[c];
	}

	// create new ones
	for (c=numToCopy; c<numVariables; c++) {
		temp[c] = new VolumeBuffer;
		if (!temp[c]) return false;
	}

	delete [] m_VolumeBuffers;
	m_VolumeBuffers = temp;
	m_NumAllocated = numVariables;
	return true;
}

// we want to be able to tell the buffer manager that we modified its
// buffer and that it should be corrected next time there is a request
// (see Bilateral Filtering in Volume/newvolumemainwindow.cpp)
void VolumeBufferManager::markBufferAsInvalid(unsigned int var)
{
	if (var >=0 && var < m_NumVariables)
		m_VolumeUpToDate[var] = false;
}

void VolumeBufferManager::invalidateBuffers()
{
	unsigned int c;
	for (c=0; c<m_NumVariables; c++) {
		m_VolumeUpToDate[c] = false;
	}
	m_GradientUpToDate = false;
}

bool VolumeBufferManager::downLoadVariable(unsigned int Variable)
{
	if (Variable>=m_NumVariables || !m_SourceManager) {
		return false;
	}
	else {
		if (m_DownLoadManager.getData(m_VolumeBuffers[Variable], m_SourceManager, 
			Variable, m_Time, 
			m_MinX, m_MinY, m_MinZ, 
			m_MaxX, m_MaxY, m_MaxZ)) {
			m_VolumeUpToDate[Variable] = true;
			return true;
		}
		else {
			return false;
		}
	}
}

bool VolumeBufferManager::downLoadGradient(unsigned int Variable)
{
	if (Variable>=m_NumVariables || !m_SourceManager) {
		return false;
	}
	else {
		if (m_DownLoadManager.getGradientData(m_GradientBuffer, m_SourceManager, 
			Variable, m_Time, 
			m_MinX, m_MinY, m_MinZ, 
			m_MaxX, m_MaxY, m_MaxZ)) {
			m_GradientUpToDate = true;
			m_GradientVariable = Variable;
			return true;
		}
		else {
			return false;
		}
	}
}


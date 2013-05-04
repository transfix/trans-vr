// ContourManager.h: interface for the ContourManager class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_CONTOURMANAGER_H__F74A5774_9FC2_471E_8EA4_E2CA1C73DBF8__INCLUDED_)
#define AFX_CONTOURMANAGER_H__F74A5774_9FC2_471E_8EA4_E2CA1C73DBF8__INCLUDED_

#include <OpenGLViewer.h>
#include <Tiling/contour.h>

class ContourManager  
{
public:
	ContourManager();
	virtual ~ContourManager();

	void setData(unsigned char* data, int dimX, int dimY, int dimZ, double aspectX, double aspectY, double aspectZ);

	void prepareContour(OpenGLViewer* viewer, double isovalue);

protected:
	void deleteDataset();

	ConDataset* m_ConDataset;
	int m_DimX;
	int m_DimY;
	int m_DimZ;
	double m_AspectX;
	double m_AspectY;
	double m_AspectZ;
};

#endif // !defined(AFX_CONTOURMANAGER_H__F74A5774_9FC2_471E_8EA4_E2CA1C73DBF8__INCLUDED_)

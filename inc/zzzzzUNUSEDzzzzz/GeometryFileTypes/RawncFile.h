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

// RawncFile.h: interface for the RawncFile class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_RAWNCFILE_H__2755A9D8_A22A_496D_BF36_C7CF6FE93F6D__INCLUDED_)
#define AFX_RAWNCFILE_H__2755A9D8_A22A_496D_BF36_C7CF6FE93F6D__INCLUDED_

#include <GeometryFileTypes/GeometryFileType.h>

///\class RawncFile RawncFile.h
///\author Anthony Thane
///\brief This is a GeometryFileType instance for reading and writing rawnc files
/// (triangles,colors,normals).
class RawncFile : public GeometryFileType  
{
public:
	virtual ~RawncFile();

	virtual Geometry* loadFile(const string& fileName);
	virtual bool checkType(const string& fileName);
	virtual bool saveFile(const Geometry* geometry, const string& fileName);

	virtual string extension() { return "rawnc"; };
	virtual string filter() { return "Rawnc files (*.rawnc)"; };

	static RawncFile ms_RawncFileRepresentative;
	static GeometryFileType* getRepresentative();

protected:
	RawncFile();
};

#endif // !defined(AFX_RAWNCFILE_H__2755A9D8_A22A_496D_BF36_C7CF6FE93F6D__INCLUDED_)

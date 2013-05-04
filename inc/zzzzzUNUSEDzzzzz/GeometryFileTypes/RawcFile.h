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

// RawcFile.h: interface for the RawcFile class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_RAWCFILE_H__F47E5524_8A5C_43D2_AFD5_8E6E5ACCB23C__INCLUDED_)
#define AFX_RAWCFILE_H__F47E5524_8A5C_43D2_AFD5_8E6E5ACCB23C__INCLUDED_

#include <GeometryFileTypes/GeometryFileType.h>

///\class RawcFile RawcFile.h
///\author Anthony Thane
///\brief This is a GeometryFileType instance for reading and writing rawc files (triangles & color).
class RawcFile : public GeometryFileType  
{
public:
	virtual ~RawcFile();

	virtual Geometry* loadFile(const string& fileName);
	virtual bool checkType(const string& fileName);
	virtual bool saveFile(const Geometry* geometry, const string& fileName);

	virtual string extension() { return "rawc"; };
	virtual string filter() { return "Rawc files (*.rawc)"; };

	static RawcFile ms_RawcFileRepresentative;
	static GeometryFileType* getRepresentative();

protected:
	RawcFile();

};

#endif // !defined(AFX_RAWCFILE_H__F47E5524_8A5C_43D2_AFD5_8E6E5ACCB23C__INCLUDED_)

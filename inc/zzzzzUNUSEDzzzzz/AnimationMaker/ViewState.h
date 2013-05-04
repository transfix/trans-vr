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

// ViewState.h: interface for the ViewState class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_VIEWSTATE_H__8B6F189E_A68D_43D0_AF61_747AFEB24F6F__INCLUDED_)
#define AFX_VIEWSTATE_H__8B6F189E_A68D_43D0_AF61_747AFEB24F6F__INCLUDED_

#include <VolumeWidget/Quaternion.h>
#include <VolumeWidget/Vector.h>
#include <stdio.h>

///\class ViewState ViewState.h
///\author Anthony Thane
///\brief The ViewState class encapsulates camera state as well as some scene state. See the
/// non-default constructor for details of the state that is stored.
class ViewState  
{
public:
	ViewState();
///\fn ViewState(const Quaternion& orientation, const Vector& target, double windowSize, double clipPlane, bool wireFrame)
///\brief Constructor
///\param orientation The orientation (rotation) of the camera
///\param target The point that the camera is looking at
///\param windowSize The window size (zoom level) of the camera
///\param clipPlane The position of the viewplane-aligned clipping plane
///\param wireFrame The wireframe state (wireframe rendering is on or off)
	ViewState(const Quaternion& orientation, const Vector& target, double windowSize, double clipPlane, bool wireFrame);
	virtual ~ViewState();

	Quaternion m_Orientation;
	double m_WindowSize;
	double m_ClipPlane;
	Vector m_Target;
	bool m_WireFrame;

///\fn void writeToFile(FILE* file)
///\brief Writes the object to an already opened file
///\param file A pointer to a FILE that is open for writing
	void writeToFile(FILE* file);
///\fn void readFromFile(FILE* file)
///\brief Reads state from an already opened file and modifies the object's state.
///\param file A pointer to a FILE that is open for reading
	void readFromFile(FILE* file);

};

// cubic interpolation
///\fn ViewState interpolate(const ViewState& a, const ViewState& b, const ViewState& c, const ViewState& d, double alpha)
///\brief Performs cubic interpolation between 4 ViewState objects. Useful for smooth animation.
///\param a A ViewState object
///\param b A ViewState object
///\param c A ViewState object
///\param d A ViewState object
///\param alpha A number between 0 and 1 that will be interpreted as the position between b and c that you want a ViewState for
///\return A ViewState object
ViewState interpolate(const ViewState& a, const ViewState& b, const ViewState& c, const ViewState& d, double alpha);
///\fn double interpolate(double a, double b, double c, double d, double alpha)
///\brief Performs cubic interpolation between 4 numbers.
///\param a A double
///\param b A double
///\param c A double
///\param d A double
///\param alpha A number between 0 and 1. This is the interpolant.
///\return A double
double interpolate(double a, double b, double c, double d, double alpha);
///\fn bool interpolate(bool a, bool b, bool c, bool d, double alpha)
///\brief Performs cubic interpolation between 4 booleans.
///\param a A bool
///\param b A bool
///\param c A bool
///\param d A bool
///\param alpha A number between 0 and 1. This is the interpolant.
///\return A bool
bool interpolate(bool a, bool b, bool c, bool d, double alpha);

// linear interpolation
///\fn ViewState interpolate(const ViewState& a, const ViewState& b, double alpha)
///\brief Performs linear interpolation between 4 ViewState objects.
///\param a A ViewState object
///\param b A ViewState object
///\param alpha A number between 0 and 1 that will be interpreted as the position between a and b that you want a ViewState for
///\return A ViewState object
ViewState interpolate(const ViewState& a, const ViewState& b, double alpha);
///\fn double interpolate(double a, double b, double alpha)
///\brief Performs linear interpolation between 2 numbers.
///\param a A double
///\param b A double
///\param alpha A number between 0 and 1. This is the interpolant.
///\return A double
double interpolate(double a, double b, double alpha);
///\fn bool interpolate(bool a, bool b, double alpha)
///\brief Performs linear interpolation between 2 booleans.
///\param a A bool
///\param b A bool
///\param alpha A number between 0 and 1. This is the interpolant.
///\return A bool
bool interpolate(bool a, bool b, double alpha);






#endif // !defined(AFX_VIEWSTATE_H__8B6F189E_A68D_43D0_AF61_747AFEB24F6F__INCLUDED_)

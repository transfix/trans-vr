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

// ViewState.cpp: implementation of the ViewState class.
//
//////////////////////////////////////////////////////////////////////

#include <AnimationMaker/ViewState.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

ViewState::ViewState()
{

}

ViewState::ViewState(const Quaternion& orientation, const Vector& target, double windowSize, double clipPlane, bool wireFrame)
: m_Orientation(orientation), m_Target(target), m_WindowSize(windowSize), m_ClipPlane(clipPlane), m_WireFrame(wireFrame)
{

}

ViewState::~ViewState()
{

}

void ViewState::writeToFile(FILE* fp)
{
	// write orientation
	fprintf(fp, "Orientation: %f %f %f %f\n", m_Orientation[0], m_Orientation[1], m_Orientation[2], m_Orientation[3]);
	// write target
	fprintf(fp, "Target: %f %f %f %f\n", m_Target[0], m_Target[1], m_Target[2], m_Target[3]);
	// write window size
	fprintf(fp, "Window Size: %f\n", (float)m_WindowSize);
	// write the clip plane
	fprintf(fp, "Clip Plane: %f\n", (float)m_ClipPlane);
	// write wireframe
	unsigned int wireFrame = (m_WireFrame?1:0);
	
	fprintf(fp, "Wireframe: %u\n", wireFrame);
}

void ViewState::readFromFile(FILE* fp)
{
	float a, b, c, d;
	// read orientation
	fscanf(fp, "Orientation: %f %f %f %f\n", &a, &b, &c, &d);
	m_Orientation.set(a, b, c, d);
	// read target
	fscanf(fp, "Target: %f %f %f %f\n", &a, &b, &c, &d);
	m_Target.set(a, b, c, d);
	// read window size
	fscanf(fp, "Window Size: %f\n", &a);
	m_WindowSize = a;
	// read the clip plane
	fscanf(fp, "Clip Plane: %f\n", &a);
	m_ClipPlane = a;
	// read wireframe
	unsigned int wireFrame;
	fscanf(fp, "Wireframe: %u\n", &wireFrame);
	m_WireFrame = (wireFrame==1?true:false);
}

// cubic interpolation
ViewState interpolate(const ViewState& a, const ViewState& b, const ViewState& c, const ViewState& d, double alpha)
{
	return ViewState(
		Quaternion::cubicInterpolate(a.m_Orientation, b.m_Orientation, c.m_Orientation, d.m_Orientation, (float)alpha),
		Vector::cubicInterpolate(a.m_Target, b.m_Target, c.m_Target, d.m_Target, (float)alpha),
		interpolate(a.m_WindowSize, b.m_WindowSize, c.m_WindowSize, d.m_WindowSize, alpha),
		interpolate(a.m_ClipPlane, b.m_ClipPlane, c.m_ClipPlane, d.m_ClipPlane, alpha),
		interpolate(a.m_WireFrame, b.m_WireFrame, c.m_WireFrame, d.m_WireFrame, alpha)
		);
}

double interpolate(double a, double b, double c, double d, double alpha)
{
	// for now, just return the linear interpolation
	return interpolate(b,c,alpha);
}

bool interpolate(bool a, bool b, bool c, bool d, double alpha)
{
	// for now, just return the linear interpolation
	return interpolate(b,c,alpha);
}

// linear interpolation
ViewState interpolate(const ViewState& a, const ViewState& b, double alpha)
{
	return ViewState(
		Quaternion::interpolate(a.m_Orientation, b.m_Orientation, (float)alpha),
		Vector::interpolate(a.m_Target, b.m_Target, (float)alpha),
		interpolate(a.m_WindowSize, b.m_WindowSize, alpha),
		interpolate(a.m_ClipPlane, b.m_ClipPlane, alpha),
		interpolate(a.m_WireFrame, b.m_WireFrame, alpha)
		);
}

double interpolate(double a, double b, double alpha) 
{
	return a*(1.0-alpha) +b*alpha;
}

bool interpolate(bool a, bool b, double alpha)
{
	return a;
}


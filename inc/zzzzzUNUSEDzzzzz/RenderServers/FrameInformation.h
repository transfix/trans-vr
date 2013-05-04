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

// FrameInformation.h: interface for the FrameInformation class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_FRAMEINFORMATION_H__64097F9B_F096_4F6F_9807_87D226A11F3D__INCLUDED_)
#define AFX_FRAMEINFORMATION_H__64097F9B_F096_4F6F_9807_87D226A11F3D__INCLUDED_

#include <ColorTable/ColorTableInformation.h>
#include <VolumeWidget/ViewInformation.h>

///\ingroup libRenderServer
///\class FrameInformation FrameInformation.h
///\brief The FrameInformation class encapsulates a camera and a transfer
///	function using the ViewInformation and ColorTableInformation classes.
///\author Anthony Thane
class FrameInformation  
{
public:
///\fn FrameInformation(const ViewInformation& viewInformation, const ColorTableInformation& colorInformation)
///\brief The class constructor.
///\param viewInformation A ViewInformation object describing the camera for a scene
///\param colorInformation A ColorTableInformation object that describes a transfer function
	FrameInformation(const ViewInformation& viewInformation, const ColorTableInformation& colorInformation);
	virtual ~FrameInformation();

///\fn const ViewInformation& getViewInformation() const
///\brief This function returns the view information.
///\return A reference to a ViewInformation object
	const ViewInformation& getViewInformation() const;
///\fn const ColorTableInformation& getColorTableInformation() const
///\brief This function returns the transfer function.
///\return A reference to a ColorTableInformation object
	const ColorTableInformation& getColorTableInformation() const;

protected:
	ViewInformation m_ViewInformation;
	ColorTableInformation m_ColorTableInformation;
};

#endif // !defined(AFX_FRAMEINFORMATION_H__64097F9B_F096_4F6F_9807_87D226A11F3D__INCLUDED_)

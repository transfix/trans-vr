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

// RenderServer.h: interface for the RenderServer class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_RENDERSERVER_H__DBA4C336_F93F_43CC_B08C_397EE68A6831__INCLUDED_)
#define AFX_RENDERSERVER_H__DBA4C336_F93F_43CC_B08C_397EE68A6831__INCLUDED_

class QPixmap;
class QWidget;
#include <RenderServers/FrameInformation.h>

///\defgroup libRenderServer Render Servers

///\ingroup libRenderServer
///\class RenderServer RenderServer.h
///\brief RenderServer is an abstract base class that represents a connection
///	to a remote process that can render images.
///\author Anthony Thane
class RenderServer  
{
public:
	RenderServer();
	virtual ~RenderServer();

///\fn virtual bool init(const char* refString, QWidget* parent = 0) = 0
///\brief This function initializes the connection to the server.
///\param refString A CORBA reference string
///\param parent a QWidget that can be used to construct other widgets
///\return A bool indicating success or failure.
	virtual bool init(const char* refString, QWidget* parent = 0) = 0;
///\fn virtual void shutdown() = 0
///\brief This function closes the connection to the server.
	virtual void shutdown() = 0;

///\fn virtual bool serverSettings(QWidget* parent = 0) = 0
///\brief This function must present some kind of setting UI to the user.
///\param parent A QWidget that can be used to construct other widgets.
///\return A bool indicating success or failure.
	virtual bool serverSettings(QWidget* parent = 0) = 0;
///\fn virtual QPixmap* renderImage(const FrameInformation& frameInformation) = 0
///\brief This function requests a rendered image from the server.
///\param frameInformation A FrameInformation object containing the current camera parameters.
///\return A pointer to a QPixmap that contains the image returned from the
///	server
	virtual QPixmap* renderImage(const FrameInformation& frameInformation) = 0;
};

#endif // !defined(AFX_RENDERSERVER_H__DBA4C336_F93F_43CC_B08C_397EE68A6831__INCLUDED_)

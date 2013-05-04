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

// RaycastRenderServer.h: interface for the RaycastRenderServer class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_RAYCASTRENDERSERVER_H__F763DCBD_D39D_4EB5_8D8D_94D704CC1EEE__INCLUDED_)
#define AFX_RAYCASTRENDERSERVER_H__F763DCBD_D39D_4EB5_8D8D_94D704CC1EEE__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <RenderServers/RenderServer.h>
#include <qstring.h>
#include <OB/CORBA.h>
#include "pvolserver.h"
class ViewInformation;
class ColorTableInformation;

///\ingroup libRenderServer
///\class RaycastRenderServer RaycastRenderServer.h
///\brief The RaycastRenderServer connects to the parallel raycasting server
///	written by Sanghun Park.
///\author Anthony Thane
class RaycastRenderServer : public RenderServer  
{
public:
	RaycastRenderServer();
	virtual ~RaycastRenderServer();

	virtual bool init(const char* refString, QWidget* parent = 0);
	virtual void shutdown();

	virtual bool serverSettings(QWidget* parent = 0);
	virtual QPixmap* renderImage(const FrameInformation& frameInformation);

protected:
	enum RenderMode { Shaded, Unshaded };

	void prepareRenderMode();
	void prepareViewInformation(const ViewInformation& viewInformation);
	void prepareColorTableInformation(const ColorTableInformation& colorTableInformation);
	void prepareImageSize();
	QPixmap* executeRender();

	void setDefaults();


	bool m_Initialized;
	RenderMode m_RenderMode;
	bool m_IsoContours;
	QString m_CurrentFile;
	unsigned int m_Width;
	unsigned int m_Height;

	CCV::PVolServer_var m_RaycastingServer;
};

#endif // !defined(AFX_RAYCASTRENDERSERVER_H__F763DCBD_D39D_4EB5_8D8D_94D704CC1EEE__INCLUDED_)

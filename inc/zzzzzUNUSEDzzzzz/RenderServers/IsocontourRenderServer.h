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

// IsocontourRenderServer.h: interface for the IsocontourRenderServer class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ISOCONTOURRENDERSERVER_H__8B79C098_98B7_4CB7_A75B_CF65B45B36D8__INCLUDED_)
#define AFX_ISOCONTOURRENDERSERVER_H__8B79C098_98B7_4CB7_A75B_CF65B45B36D8__INCLUDED_

#include <RenderServers/RenderServer.h>
#include <OB/CORBA.h>
#include "cr.h"

///\ingroup libRenderServer
///\class IsocontourRenderServer IsocontourRenderServer.h
///\brief The IsoContourRenderSever connects to the parallel isocontour render
///	server written by Xiaoyu Zhang.
///\author Anthony Thane
class IsocontourRenderServer : public RenderServer  
{
public:
	IsocontourRenderServer();
	virtual ~IsocontourRenderServer();

	virtual bool init(const char* refString, QWidget* parent = 0);
	virtual void shutdown();

	virtual bool serverSettings(QWidget* parent = 0);
	virtual QPixmap* renderImage(const FrameInformation& frameInformation);

protected:
	void prepareViewInformation(const ViewInformation& viewInformation);
	double prepareIsocontourInformation(const ColorTableInformation& colorTableInformation);
	QPixmap* executeRender(double value);

	void setDefaults();


	bool m_Initialized;
	QString m_CurrentFile;

	ccv::CRServer_var m_IsocontourServer;
};

#endif // !defined(AFX_ISOCONTOURRENDERSERVER_H__8B79C098_98B7_4CB7_A75B_CF65B45B36D8__INCLUDED_)

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

// TextureRenderServer.h: interface for the TextureRenderServer class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_TEXTURERENDERSERVER_H__F6336629_FD6D_401F_BC71_D80DE11CFB37__INCLUDED_)
#define AFX_TEXTURERENDERSERVER_H__F6336629_FD6D_401F_BC71_D80DE11CFB37__INCLUDED_

#include <RenderServers/RenderServer.h>
#include <OB/CORBA.h>
#include "3DTex.h"

///\class TextureRenderServer TextureRenderServer.h
///\brief The TextureRenderServer connects to the 3D texture-based render
///	server written by Sangmin Park & Bongjune Kwon.
///\author Anthony Thane
class TextureRenderServer : public RenderServer  
{
public:
	TextureRenderServer();
	virtual ~TextureRenderServer();

	virtual bool init(const char* refString, QWidget* parent = 0);
	virtual void shutdown();

	virtual bool serverSettings(QWidget* parent = 0);
	virtual QPixmap* renderImage(const FrameInformation& frameInformation);

protected:
	void prepareViewInformation(const ViewInformation& viewInformation);
	void prepareColorTableInformation(const ColorTableInformation& colorTableInformation);
	QPixmap* executeRender();

	void setDefaults();


	bool m_Initialized;
	QString m_CurrentFile;

	CCVSM::PVolServerSM_var m_TextureServer;
};

#endif // !defined(AFX_TEXTURERENDERSERVER_H__F6336629_FD6D_401F_BC71_D80DE11CFB37__INCLUDED_)

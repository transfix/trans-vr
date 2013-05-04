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

// TextureRenderServer.cpp: implementation of the TextureRenderServer class.
//
//////////////////////////////////////////////////////////////////////

#include <RenderServers/TextureRenderServer.h>
#include <RenderServers/Corba.h>
#include <OB/CORBA.h>
#include "3DTex.h"
#include <RenderServers/textureserversettingsdialogimpl.h>
#include <RenderServers/TransferArray.h>
#include <VolumeWidget/Quaternion.h>
#include <VolumeWidget/Matrix.h>
#include <qlistbox.h>
#include <qimage.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

TextureRenderServer::TextureRenderServer()
{
	setDefaults();
}

TextureRenderServer::~TextureRenderServer()
{
	shutdown();
}

bool TextureRenderServer::init(const char* refString, QWidget* parent)
{
	// get orb
	CORBA::ORB*	orb = Corba::getOrb();
	if (!orb) {
		return false;
	}

	// get texture server object
	try {
		CORBA::String_var str((const char*)refString);

		CORBA::Object_var obj = orb->string_to_object(str);
		m_TextureServer = CCVSM::PVolServerSM::_narrow(obj);
	}
	catch(CORBA::SystemException&) {
		// failure 
		m_Initialized = false;
		return false;
	}

	if (serverSettings(parent)) { // success
		m_Initialized = true;
		return true;
	}
	else { // error occurred while preparing settings
		m_Initialized = false;
		return false;
	}
}

void TextureRenderServer::shutdown()
{
	//delete m_TextureServer;
	m_TextureServer = 0;
	m_Initialized = false;
	m_CurrentFile = QString("");
}


bool TextureRenderServer::serverSettings(QWidget* parent)
{
	if (!m_TextureServer) {
		return false;
	}

	TextureServerSettingsDialog dialog(parent);

	CCVSM::FileList_var list;
	try {
		list = m_TextureServer->getConfigFiles();
	}
	catch (CORBA::SystemException& e) {
		// error getting file list
		qDebug("Error getting file list.  Reason:");
		qDebug(e.reason());
		return false;
	}

	// fill the file list box
	unsigned int c;
	for (c=0; c<list->length(); c++) {
		QString item((*list)[c]);
		dialog.m_FileListBox->insertItem(item);
		if (item==m_CurrentFile) {
			dialog.m_FileListBox->setCurrentItem(c);
		}
	}

	if (dialog.m_FileListBox->currentItem()<0) {
		dialog.m_FileListBox->setCurrentItem(0);
	}

	if (dialog.exec()==QDialog::Accepted) {
		// load the file if its different from m_CurrentFile
		if (dialog.m_FileListBox->currentText()!=m_CurrentFile) {
			m_CurrentFile = dialog.m_FileListBox->currentText();
			try {
				m_TextureServer->loadData(m_CurrentFile);
			}
			catch(CORBA::SystemException& e) {
				// error loading file
				qDebug("Error loading file.  Reason:");
				qDebug(e.reason());
				return false;
			}
		}
	}

	return true;
}

QPixmap* TextureRenderServer::renderImage(const FrameInformation& frameInformation)
{
	if (!m_TextureServer|| !m_Initialized) {
		// we aren't ready to render images
		return 0;
	}
	QPixmap* result = 0;

	try {

		// prepare view information
		prepareViewInformation(frameInformation.getViewInformation());

		// prepare colortable information
		prepareColorTableInformation(frameInformation.getColorTableInformation());

		// Render the image
		result = executeRender();

	}
	catch (CORBA::SystemException& e) {
		result = 0;
		// error rendering image
		qDebug("Error rendering image.  Reason:");
		qDebug(e.reason());
	}
	return result;
}

void TextureRenderServer::prepareViewInformation(const ViewInformation& viewInformation)
{
	Quaternion orientation(viewInformation.getOrientation());
	Matrix m = orientation.buildMatrix();

	CCVSM::SRotationMatrix   CCVQuatMat;
	
	unsigned int i,j;
	for (i=0; i<4; i++) {
		for (j=0; j<4; j++) {
			CCVQuatMat.RotMat[i][j] = m.getMatrix()[j*4+i];
		}
	}

	// 0 means server will not reload file
	// I think this needs to be called before we set orientation
	m_TextureServer->setFileNumber(0);
	m_TextureServer->setRotationMatrix(CCVQuatMat);
	m_TextureServer->setViewFrustum(viewInformation.getWindowSize()*2, viewInformation.getClipPlane()/*5.0*/, 0.5, 0.5);


}

void TextureRenderServer::prepareColorTableInformation(const ColorTableInformation& colorTableInformation)
{
	TransferArray transferArray(16);
	transferArray.buildFromColorTable(colorTableInformation);

	// prepare the texturebased version of the transfer function
	CCVSM::SeqSTFNNodes transfer;
	transfer.length(transferArray.getNumElements());

	unsigned int c;
	double red, green, blue, alpha, position;

	for (c=0; c<transferArray.getNumElements(); c++) {
		position = transferArray[c].m_Position;
		alpha = transferArray[c].m_Alpha;
		red = transferArray[c].m_Red;
		green = transferArray[c].m_Green;
		blue = transferArray[c].m_Blue;
		transfer[c].Density = (int)(position*255);
		transfer[c].R = (int)(red*255);
		transfer[c].G = (int)(green*255);
		transfer[c].B = (int)(blue*255);
		transfer[c].A = (int)(alpha*255);
	}		

	// Send the transfer function to server	
	m_TextureServer->setTransferFunctionNodes(transfer);
}

QPixmap* TextureRenderServer::executeRender()
{
	// Render the image
	CCVSM::Image *pImage = m_TextureServer->getRenderedImage();
	QImage image( pImage->dx, pImage->dy, 32); 

	unsigned int x,y;
	for (y=0; y<pImage->dy; y++) {
		for (x=0; x<pImage->dx; x++) {
			image.setPixel(x,y,qRgb(pImage->img[(y)*pImage->dx*3 + (x)*3 + 0],
									pImage->img[(y)*pImage->dx*3 + (x)*3 + 1],
									pImage->img[(y)*pImage->dx*3 + (x)*3 + 2]));
		}
	}

	return new QPixmap(image);
}

void TextureRenderServer::setDefaults()
{
	m_TextureServer = 0;
	m_Initialized = false;
	m_CurrentFile = QString("");
}


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

// RaycastRenderServer.cpp: implementation of the RaycastRenderServer class.
//
//////////////////////////////////////////////////////////////////////

#include <RenderServers/RaycastRenderServer.h>
#include <RenderServers/Corba.h>
#include <OB/CORBA.h>
#include "pvolserver.h"
#include <RenderServers/raycastserversettingsdialogimpl.h>
#include <RenderServers/TransferArray.h>
#include <VolumeWidget/Vector.h>
#include <VolumeWidget/Quaternion.h>
#include <RenderServers/FrameInformation.h>
#include <math.h>
#include <qlineedit.h>
#include <qcheckbox.h>
#include <qradiobutton.h>
#include <qlistbox.h>
#include <qimage.h>
#include <qdatetime.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

RaycastRenderServer::RaycastRenderServer()
{
	setDefaults();
}

RaycastRenderServer::~RaycastRenderServer()
{
	shutdown();
}

bool RaycastRenderServer::init(const char* refString, QWidget* parent)
{
	// get orb
	CORBA::ORB*	orb = Corba::getOrb();
	if (!orb) {
		return false;
	}

	// get raycast server object
	try {
		CORBA::String_var str((const char*)refString);

		CORBA::Object_var obj = orb->string_to_object(str);
		m_RaycastingServer = CCV::PVolServer::_narrow(obj);
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

void RaycastRenderServer::shutdown()
{
	//delete m_RaycastingServer;
	m_RaycastingServer = 0;
	m_Initialized = false;
	m_CurrentFile = QString("");
}

bool RaycastRenderServer::serverSettings(QWidget* parent)
{
	if (!m_RaycastingServer) {
		return false;
	}

	RaycastServerSettingsDialog dialog(parent);
	dialog.m_WidthEditBox->setText(QString::number(m_Width));
	dialog.m_HeightEditBox->setText(QString::number(m_Height));
	dialog.m_IsosurfacingBox->setChecked(m_IsoContours);
	dialog.m_ShadedButton->setChecked((m_RenderMode==Shaded?true:false));
	dialog.m_UnshadedButton->setChecked((m_RenderMode==Unshaded?true:false));

	CCV::FileList_var list;
	try {
		list = m_RaycastingServer->getConfigFiles();
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
				QTime time;
				time.start();
				m_RaycastingServer->loadData(m_CurrentFile);
				qDebug("Elapsed time to load file: %d.", time.elapsed());
			}
			catch(CORBA::SystemException& e) {
				// error loading file
				qDebug("Error loading file.  Reason:");
				qDebug(e.reason());
				return false;
			}
		}

		bool ok;
		m_Height = dialog.m_HeightEditBox->text().toInt(&ok);
		m_Width = dialog.m_HeightEditBox->text().toInt(&ok);

		m_IsoContours = dialog.m_IsosurfacingBox->isChecked();
		if (dialog.m_ShadedButton->isChecked()) {
			m_RenderMode = Shaded;
		}
		else {
			m_RenderMode = Unshaded;
		}

	}

	return true;
}

static inline int maxOfThree(int n1, int n2, int n3) {
	int max = (n1>n2?n1:n2);
	return (max>n3?max:n3);
}

QPixmap* RaycastRenderServer::renderImage(const FrameInformation& frameInformation)
{
	if (!m_RaycastingServer || !m_Initialized) {
		// we aren't ready to render images
		return 0;
	}
	QPixmap* result = 0;

	try {

		// prepare render mode
		prepareRenderMode();

		// prepare view information
		prepareViewInformation(frameInformation.getViewInformation());

		// prepare colortable information
		prepareColorTableInformation(frameInformation.getColorTableInformation());

		// set image size
		prepareImageSize();

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

void RaycastRenderServer::setDefaults()
{
	m_Initialized = false;
	m_RaycastingServer = false;
	m_RenderMode = Shaded;
	m_IsoContours = false;
	m_Width = 512;
	m_Height = 512;
	m_CurrentFile = QString("");

}

void RaycastRenderServer::prepareRenderMode()
{
	// set the render mode
	int rendermode = 0;
	if (m_RenderMode==Shaded) {
		rendermode |= 0x1;
	}
	if (m_IsoContours) {
		rendermode |= 0x2;
	}
	if (m_RenderMode==Unshaded) {
		rendermode |= 0x4;
	}
	m_RaycastingServer->setRenderingMode(rendermode);
}

void RaycastRenderServer::prepareViewInformation(const ViewInformation& viewInformation)
{
	// get data info
	CCV::DataInfo datainfo = m_RaycastingServer->getDataInfo();

	// prepare view information
	Vector EyeDirection(0.0f, 0.0f, 1.0f, 0.0f);
	Vector EyeUp(0.0f, 1.0f, 0.0f, 0.0f);
	Vector TowardsLight(-1, 1, 3, 0.0f); // Light
	Vector EyePosition;
	long max = maxOfThree(datainfo.dim[0], datainfo.dim[1], datainfo.dim[2])-1;
	if (viewInformation.isPerspective()) {
		float distance, fov = viewInformation.getFov();
		float windowSize = viewInformation.getWindowSize();


		distance = windowSize / 2.0f / (float)tan( fov/2.0f );
		EyePosition.set(0.0f, 0.0f, distance*(float)max, 1.0);

	}
	else {
		EyePosition.set(0.0f, 0.0f, 800.0f, 1.0);

	}
	Quaternion orientation(viewInformation.getOrientation());
	EyeDirection = orientation.applyRotation(EyeDirection);
	EyeUp = orientation.applyRotation(EyeUp);
	TowardsLight = orientation.applyRotation(TowardsLight);
	EyePosition = orientation.applyRotation(EyePosition);
	Vector target(viewInformation.getTarget());
	target[3] = 0;
	EyePosition+=(target*(float)max);

	CCV::ViewParam myparam;
	myparam.fov = viewInformation.getFov()*180.0/(3.1415925);
	myparam.vec[0] = EyeDirection[0];
	myparam.vec[1] = EyeDirection[1];
	myparam.vec[2] = EyeDirection[2];//-EyeDirection.getZ();
	myparam.pos[0] = EyePosition[0];
	myparam.pos[1] = EyePosition[1];
	myparam.pos[2] = EyePosition[2];//-EyePosition.getZ();
	myparam.up[0] = EyeUp[0];
	myparam.up[1] = EyeUp[1];
	myparam.up[2] = EyeUp[2];//-EyeUp.getZ();
	myparam.perspective = viewInformation.isPerspective();


	// set window size in object space
	CCV::WinSize ws;
	ws[0] = viewInformation.getWindowSize()*max; ws[1] = viewInformation.getWindowSize()*(float)max;
	m_RaycastingServer->setWindowSize(ws);
	

	m_RaycastingServer->setViewParam(myparam);
}

double MapTo(double val, double min, double max)
{

	val*=max-min;
	val+=min;

	return val;
}

void RaycastRenderServer::prepareColorTableInformation(const ColorTableInformation& colorTableInformation)
{
	TransferArray transferArray(16);
	transferArray.buildFromColorTable(colorTableInformation);

	// prepare the raycaster version of the transfer function
	CCV::DataInfo datainfo= m_RaycastingServer->getDataInfo();
	double min = datainfo.ext[0];
	double max = datainfo.ext[1];

	CCV::DenTransfer transfer;
	transfer.length(transferArray.getNumElements());
	unsigned int c;
	double red,green,blue,alpha,position;
	int density;
	for (c=0; c<transferArray.getNumElements(); c++) {
		position = transferArray[c].m_Position;
		density = (int)MapTo(position, min, max);
		alpha = transferArray[c].m_Alpha;
		red = transferArray[c].m_Red;
		green = transferArray[c].m_Green;
		blue = transferArray[c].m_Blue;
		transfer[c].den = density;
		transfer[c].r = (int)(red*255);
		transfer[c].g = (int)(green*255);
		transfer[c].b = (int)(blue*255);
		transfer[c].alpha = (int)(alpha*255);
	}

	// Send the transfer function to server
	m_RaycastingServer->setDenTransfer(transfer);
	
	if (m_IsoContours && 
		colorTableInformation.getIsocontourMap().GetSize()>0) {
		CCV::ContourInfo con_info;
		con_info.val = MapTo(colorTableInformation.getIsocontourMap().GetPositionOfIthNode(0), min, max);

		// fully opaque
		con_info.opacity = 1.0;
		// white specular
		//con_info.sr = 255;
		//con_info.sg = 255;
		//con_info.sb = 255;
		con_info.sr = (CORBA::Long)(colorTableInformation.getIsocontourMap().GetRed(0)*255.);
		con_info.sg = (CORBA::Long)(colorTableInformation.getIsocontourMap().GetGreen(0)*255.);
		con_info.sb = (CORBA::Long)(colorTableInformation.getIsocontourMap().GetBlue(0)*255.);
		// no ambient
		con_info.ar = 0;
		con_info.ag = 0;
		con_info.ab = 0;
		// white diffuse for now
		//con_info.dr = 255;
		//con_info.dg = 255;
		//con_info.db = 255;
		con_info.dr = (CORBA::Long)(colorTableInformation.getIsocontourMap().GetRed(0)*255.);
		con_info.dg = (CORBA::Long)(colorTableInformation.getIsocontourMap().GetGreen(0)*255.);
		con_info.db = (CORBA::Long)(colorTableInformation.getIsocontourMap().GetBlue(0)*255.);

		// there's already one contour set for the server I think
		m_RaycastingServer->setContourInfo(con_info, 0);
	}

}

void RaycastRenderServer::prepareImageSize()
{
	// set image size
	CCV::WinSize is;
	is[0] = m_Width; is[1] = m_Height;
	m_RaycastingServer->setImageSize(is);
}

QPixmap* RaycastRenderServer::executeRender()
{
	// Render the image
	QTime time;
	time.start();
	CCV::Image *p_img = m_RaycastingServer->render();
	qDebug("Elapsed time: %d.", time.elapsed());
	QImage image(m_Width, m_Height, 32); 

	unsigned int x,y;
	for (y=0; y<m_Height; y++) {
		for (x=0; x<m_Width; x++) {
			/*
			image.setPixel(x,y,qRgb(p_img->img[y*m_Width*3 + (m_Width-x-1)*3 + 0],
									p_img->img[y*m_Width*3 + (m_Width-x-1)*3 + 1],
									p_img->img[y*m_Width*3 + (m_Width-x-1)*3 + 2]));
									*/
			image.setPixel(x,y,qRgb(p_img->img[y*m_Width*3 + (x)*3 + 0],
									p_img->img[y*m_Width*3 + (x)*3 + 1],
									p_img->img[y*m_Width*3 + (x)*3 + 2]));
		}
	}

	return new QPixmap(image);
}



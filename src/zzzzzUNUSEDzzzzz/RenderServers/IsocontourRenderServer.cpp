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

// IsocontourRenderServer.cpp: implementation of the IsocontourRenderServer class.
//
//////////////////////////////////////////////////////////////////////

#include <RenderServers/IsocontourRenderServer.h>
#include <RenderServers/Corba.h>
#include <RenderServers/textureserversettingsdialogimpl.h>
#include <qlistbox.h>
#include <qimage.h>
#include <VolumeWidget/Matrix.h>
#include <VolumeWidget/PerspectiveView.h>
#include <OB/CosNaming.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

IsocontourRenderServer::IsocontourRenderServer()
{
	setDefaults();
}

IsocontourRenderServer::~IsocontourRenderServer()
{
	shutdown();
}

bool IsocontourRenderServer::init(const char* refString, QWidget* parent)
{
	// get orb
	CORBA::ORB*	orb = Corba::getOrb();
	if (!orb) {
		return false;
	}

	// name on name service "contourrenderserver"

	// get texture server object
	try {
		CORBA::Object_var nameObj = orb->string_to_object(refString);
		CosNaming::NamingContext_var nc = CosNaming::NamingContext::_narrow(nameObj);
		CosNaming::Name name;
		name.length(1);
		name[0].id = CORBA::string_dup("ContourRenderServer");
		CORBA::Object_var obj = nc->resolve(name);
		m_IsocontourServer = ccv::CRServer::_narrow(obj);


		//CORBA::String_var str((const char*)refString);

		//CORBA::Object_var obj = orb->string_to_object(str);
		//m_IsocontourServer = ccv::CRServer::_narrow(obj);
	}
	catch(CORBA::Exception&) {
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

void IsocontourRenderServer::shutdown()
{
	setDefaults();
}

bool IsocontourRenderServer::serverSettings(QWidget* parent)
{
	if (!m_IsocontourServer) {
		return false;
	}

	TextureServerSettingsDialog dialog(parent);

	ccv::CRFileList_var list;
	try {
		list = m_IsocontourServer->getFileList();
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
		QString item((*list)[c].name);
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
				c = dialog.m_FileListBox->currentItem();
				m_IsocontourServer->loadData(m_CurrentFile, (*list)[c].type);
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

QPixmap* IsocontourRenderServer::renderImage(const FrameInformation& frameInformation)
{
	if (!m_IsocontourServer || 
		!m_Initialized || 
		frameInformation.getColorTableInformation().getIsocontourMap().GetSize()<1) {
		// we aren't ready to render images
		return 0;
	}
	QPixmap* result = 0;

	try {

		// prepare view information
		prepareViewInformation(frameInformation.getViewInformation());

		double value = prepareIsocontourInformation(frameInformation.getColorTableInformation());

		// Render the image
		result = executeRender(value);

	}
	catch (CORBA::SystemException& e) {
		result = 0;
		// error rendering image
		qDebug("Error rendering image.  Reason:");
		qDebug(e.reason());
	}

	return result;
}

static inline float maxOfThree(float n1, float n2, float n3) {
	float max = (n1>n2?n1:n2);
	return (max>n3?max:n3);
}

void IsocontourRenderServer::prepareViewInformation(const ViewInformation& viewInformation)
{
	// get dataset information
	ccv::VolumeInfo info = m_IsocontourServer->getDataInfo();

	// undo some crazy stuff at the server
	// see the isocontour server for why these instructions are necessary
	double bsize = maxOfThree(info.max[0]-info.min[0],
		info.max[1]-info.min[1],
		info.max[2]-info.min[2]);
	Vector eye((info.max[0]+info.min[0])/2.0,
		(info.max[1]+info.min[1])/2.0,
		(info.max[2]+info.min[2])/2.0 - bsize*1.732,
		0);

	// translate
	Matrix matrix = Matrix::translation(eye);

	// rotate
	matrix.postMultiplication(Matrix::rotationX(3.1415925f));

	// get a perspective modelview equivalent to the viewinformation
	PerspectiveView view(viewInformation);
	// the server has a fixed 60 degree field of view
	view.setFieldOfView(60.0f*(3.1415925f)/180.0f);
	view.SetWindowSize(view.GetWindowSize()*bsize);

	matrix.postMultiplication(view.getModelViewMatrix());

	
	//matrix.preMultiplication(Matrix::scale(1.0f,1.0f,-1.0f));

	// scale the data
	/*
	float scale = maxOfThree(info.max[0]-info.min[0],
		info.max[1]-info.min[1],
		info.max[2]-info.min[2]);
	matrix.preMultiplication(Matrix::scale(1.0f/scale,1.0f/scale,1.0f/scale));*/

	// center the data
	matrix.postMultiplication(Matrix::translation(-(info.max[0]+info.min[0])/2.0,
		-(info.max[1]+info.min[1])/2.0,
		-(info.max[2]+info.min[2])/2.0));
	
	// convert to the server's matrix
	ccv::TMatrix ccvMatrix;
	ccvMatrix.length(16);
	unsigned int c;
	for (c=0; c<4; c++) {
		ccvMatrix[c*4+0] = matrix.getMatrix()[0*4+c];
		ccvMatrix[c*4+1] = matrix.getMatrix()[1*4+c];
		ccvMatrix[c*4+2] = matrix.getMatrix()[2*4+c];
		ccvMatrix[c*4+3] = matrix.getMatrix()[3*4+c];
	}

	// send the matrix to the server
	m_IsocontourServer->setTransformation(ccvMatrix); 
}

double IsocontourRenderServer::prepareIsocontourInformation(const ColorTableInformation& colorTableInformation)
{
	// get dataset information
	ccv::VolumeInfo info = m_IsocontourServer->getDataInfo();

	// we are sure to have at least one element, we checked earlier
	double value = colorTableInformation.getIsocontourMap().GetPositionOfIthNode(0);

	return value*(info.isomax-info.isomin)+info.isomin;

}

QPixmap* IsocontourRenderServer::executeRender(double value)
{
	// Render the image
	ccv::Image *ccvImage = m_IsocontourServer->render(value);
	QImage image( ccvImage->dx, ccvImage->dy, 32); 

	unsigned int x,y;
	for (y=0; y<ccvImage->dy; y++) {
		for (x=0; x<ccvImage->dx; x++) {
			image.setPixel(x,y,qRgb(ccvImage->img[(ccvImage->dy-y-1)*ccvImage->dx*3 + (x)*3 + 0],
									ccvImage->img[(ccvImage->dy-y-1)*ccvImage->dx*3 + (x)*3 + 1],
									ccvImage->img[(ccvImage->dy-y-1)*ccvImage->dx*3 + (x)*3 + 2]));
		}
	}

	return new QPixmap(image);
}

void IsocontourRenderServer::setDefaults()
{
	m_IsocontourServer = 0;
	m_Initialized = false;
	m_CurrentFile = QString("");
}


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

// DataCutterSource.h: interface for the DataCutterSource class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_DATACUTTERSOURCE_H__B27AA6FF_F544_4CEC_9F81_4EAB1EA1815D__INCLUDED_)
#define AFX_DATACUTTERSOURCE_H__B27AA6FF_F544_4CEC_9F81_4EAB1EA1815D__INCLUDED_

#include <VolumeFileTypes/VolumeSource.h>
#include <dataCutterClient/message.h>
#include <qstring.h>
#include <qwidget.h>

#include <Q3Socket>

//class Q3Socket;
//class QStringList;

#define BUFSIZE 10000

///\class DataCutterSource DataCutterSource.h
///\author Anthony Thane
///\brief The DataCutterSource is a VolumeSource that connects to a Data
/// Cutter server (see 
/// <a href="http://datacutter.osu.edu">http://datacutter.osu.edu</a>).
class DataCutterSource : public VolumeSource  
{
public:
///\fn DataCutterSource(QWidget* parent)
///\brief The constructor
///\param parent A QWidget instance that will by used by the class to init
/// it's GUI. (see QWidget's description for details)
	DataCutterSource(QWidget* parent);
	virtual ~DataCutterSource();


private:
	//int sockfd;
	Q3Socket* sockfd;
	Message *pmess;
	HelloMessage *phm;
	HelloMessageAck *phma;
	GetDSDetailsMessage *pgetdsd;
	GetDSDetailsMessageAck *pgetdsda;
	GetDSMessage *pgetds;
	GetDSMessageAck *pgetdsa;
	ErrorMessage *perr;
	ByeMessage *pbye;
	
	uchar buffer[BUFSIZE];
	
	/* input dataset name */
	char dsname[255];
	
	/* input dataset spec */
	DSDetails *spec;

	/* the dataset spec for the current file */
	DSDetails *currentSpec;
	
	// the parent widget of the progress dialog
	QWidget* m_ProgressParent;
	
public:	
	
	QStringList setSockfd(Q3Socket* sfd);
	void setSpec(DSDetails *sp);
	void setDSName(const char *ldsname);
	
	/* request for dataset and write to filename */
	//void request(); 
	
	void requestHello();
	void requestGetDSDetails();
	void requestGetDS();
	void requestBye();
	void requestError();
	
	QStringList processHelloAck();
	void processGetDSDetailsAck();
	void processGetDSAck(char* data, unsigned int buffSize);
	void processBye();
	void processError();
	
	void recvMessage();
	void sendMessage();
	
	void freeAll();

	virtual void resetError();

	virtual void fillData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax,
		uint xDim, uint yDim, uint zDim, uint variable, uint timeStep);

	virtual void fillThumbnail(char* data, 
		unsigned int xDim, unsigned int yDim, unsigned int zDim, uint variable, uint timeStep);
	
	virtual unsigned int getVariableType(unsigned int variable) const;
};

#endif // !defined(AFX_DATACUTTERSOURCE_H__B27AA6FF_F544_4CEC_9F81_4EAB1EA1815D__INCLUDED_)

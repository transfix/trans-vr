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

// DataCutterSource.cpp: implementation of the DataCutterSource class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeFileTypes/DataCutterSource.h>
//#include <iostream>
//#include <fstream>
//#include <qsocket.h>
#include <Q3Socket>
#include <Q3ProgressDialog>
//#include <qstringlist.h>
#include <qdatetime.h>
#include <qprogressdialog.h>
#include <qapplication.h>

//#define ZCOMPRESSIONACTIVE
//using namespace std;

#ifdef ZCOMPRESSIONACTIVE

#include <zlib.h>

// zlib required functions
void* DCalloc(voidpf opaque, uInt items, uInt size);
void  DCfree(voidpf opaque, voidpf address);

void* DCalloc_func(voidpf opaque, uInt items, uInt size)
{
	void * memory = malloc(items*size);
	if (memory) {
		return memory;
	}
	else {
		return Z_NULL;
	}
}

void  DCfree_func(voidpf opaque, voidpf address)
{
	if (address) {
		free(address);
	}
}

#endif


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

DataCutterSource::DataCutterSource(QWidget* parent)
{
	sockfd=NULL;

	pmess=NULL;
	phm=NULL;
	phma=NULL;
	pgetdsd=NULL;
	pgetdsda=NULL;
	pgetds=NULL;
	pgetdsa=NULL;
	perr=NULL;
	pbye=NULL;

	strcpy(dsname, "NotSet!");
	currentSpec=NULL;

	m_ProgressParent = parent;

	setDefaults();
}

DataCutterSource::~DataCutterSource()
{
	if (!error()) {
		requestBye();
		recvMessage();
		if (perr!=NULL)
			processError();
		else
			processBye();
	}
	freeAll();
	sockfd->close();
	//delete sockfd;
}

QStringList DataCutterSource::setSockfd(Q3Socket* sfd)
{
	sockfd=sfd;

	// say hi and get dataset list
	requestHello();
	recvMessage();
	if (perr!=NULL) {
		processError();
		return QStringList();
	}
	else {
		return processHelloAck();
	}
	 
}

void DataCutterSource::setSpec(DSDetails *sp)
{
	spec=sp;
}

void DataCutterSource::setDSName(const char *ldsname)
{
	strcpy(dsname, ldsname);

	// request the dataset details
	requestGetDSDetails();
	recvMessage();
	if (perr!=NULL)
		processError();
	else
		processGetDSDetailsAck();
}

	
	/* request for dataset and write to filename */
/*
void DataCutterSource::request()
{
	requestHello();
	recvMessage();
	processHelloAck();


	requestGetDSDetails();
	recvMessage();
	processGetDSDetailsAck();


	requestGetDS();
	recvMessage();
	processGetDSAck();

	requestBye();
	recvMessage();
	processBye();


	return;
}
*/
	
void DataCutterSource::requestHello()
{
	phm=new HelloMessage();
	memset(buffer,'A',BUFSIZE);

	phm->serialize(buffer);
	sendMessage();
}

void DataCutterSource::requestGetDSDetails()
{
	pgetdsd=new GetDSDetailsMessage();

#ifdef DEBUG2
	cout<<"going to request for "<<dsname<<" details"<<endl;
#endif

	pgetdsd->setName((uchar *) dsname);
	pgetdsd->serialize(buffer);
	sendMessage();
}

void DataCutterSource::requestGetDS()
{
	pgetds=new GetDSMessage();

#ifdef DEBUG2
	cout<<"requesting ds!"<<endl;
#endif


	pgetds->setName((uchar *)dsname);
	pgetds->setSpec(spec);

	pgetds->serialize(buffer);
	sendMessage();
}

void DataCutterSource::requestBye()
{
	pbye=new ByeMessage();

#ifdef DEBUG2
	cout<<"requesting bye!"<<endl;
#endif

	pbye->serialize(buffer);
	sendMessage();
}

void DataCutterSource::requestError()
{

}

	
QStringList DataCutterSource::processHelloAck()
{
	int i;
	int n;
	QStringList DataSets;
	char lname[100];
	char ldesc[200];

#ifdef DEBUG2
	cout<<"processing hello ack"<<endl;
#endif

	phma->reset();
	n=phma->getNsets();

	for (i=0;i<n;i++)
		{
		phma->getNext((uchar*)lname,(uchar*)ldesc);
		DataSets.append(lname);
		//cout<<"\nName="<<lname<<endl;
		//cout<<"\nDescription="<<ldesc<<endl;
		}
	return DataSets;
}

void DataCutterSource::processGetDSDetailsAck()
{
	//cout<<"\nName: "<<pgetdsda->getName()<<endl;
	//cout<<"\nDetails: "<<endl;
	(pgetdsda->getDetails())->display();

	// save the dataset details
	if (currentSpec) delete currentSpec;
	currentSpec = new DSDetails(*(pgetdsda->getDetails()));

	unsigned int min[3];
	unsigned int max[3];
	unsigned int dim[3];
	currentSpec->getMin(min);
	currentSpec->getMax(max);
	currentSpec->getDim(dim);
	m_DimX = dim[0];
	m_DimY = dim[1];
	m_DimZ = dim[2];

	m_MaxX = max[0];
	m_MaxY = max[1];
	m_MaxZ = max[2];

	m_MinX = min[0];
	m_MinY = min[1];
	m_MinZ = min[2];

	m_MinT = 0.0f;
	m_MaxT = 0.0f;

	m_NumVariables = 1;
	m_NumTimeSteps = 1;

	//cout<<endl;
}

void DataCutterSource::processGetDSAck(char* data, unsigned int buffSize)
{
#ifdef DEBUG2
	//cout<<"processing getds ack"<<endl;
#endif

	//cout<<"name of dataset = "<<pgetdsa->getName()<<endl;
	//cout<<"type of dataset = "<<(int) pgetdsa->getDataType()<<endl;

#ifdef ZCOMPRESSIONACTIVE
	// basic initialization of a_stream object
	z_stream zs;
	memset(&zs, 0, sizeof(zs));
	zs.zalloc = DCalloc_func;
	zs.zfree = DCfree_func;
	zs.opaque = 0;
	zs.data_type = Z_BINARY;
#endif

	//cout<<"Going to receive data from server... and writing to file "<<filename<<endl;


	uchar *pbuffer=buffer;
	int nbytesleft=pgetdsa->getNBytes();
	int ntotalbytes=pgetdsa->getNBytes();
	int nbytesread;
#ifdef ZCOMPRESSIONACTIVE
	int nuncompressedsize = buffSize;
	int zlibReturn;
	inflateInit(&zs);
#endif

	
	// create the progressbar
	Q3ProgressDialog* progressDialog = new Q3ProgressDialog("Downloading data from the DataCutter server.", 
		0, ntotalbytes, m_ProgressParent, "Downloading Data", true);

	while (sockfd->bytesAvailable()==0);
	nbytesread=sockfd->readBlock((char*)pbuffer,(BUFSIZE<nbytesleft ? BUFSIZE : nbytesleft));

#ifdef ZCOMPRESSIONACTIVE
	zs.next_out = (unsigned char*)data;
	zs.avail_out = nuncompressedsize;
#endif

	while(nbytesread>0)
		{
		nbytesleft -= nbytesread;

#ifdef ZCOMPRESSIONACTIVE
		// decompress from data to pbufer
		zs.next_in = pbuffer;
		zs.avail_in = nbytesread;
		zlibReturn = inflate(&zs, Z_NO_FLUSH);
		if (zlibReturn==Z_STREAM_END) {
			qDebug("Got to end of stream");
		}
		else if (zlibReturn != Z_OK) {
			qDebug("Error");
		}
#else
		// copy from data to pbuffer
		memcpy(data, pbuffer, nbytesread);
#endif

		progressDialog->setProgress(ntotalbytes-nbytesleft);
		qApp->processEvents();

		data+=nbytesread;
		if (nbytesleft <= 0) break;
		while (sockfd->bytesAvailable()==0);
		nbytesread=sockfd->readBlock((char*)pbuffer,(BUFSIZE<nbytesleft ? BUFSIZE : nbytesleft));
		}


	if (sockfd->bytesAvailable()!=0) 
		nbytesread=sockfd->readBlock((char*)pbuffer,BUFSIZE);

	delete progressDialog;

#ifdef ZCOMPRESSIONACTIVE
	inflateEnd(&zs);
#endif


}

void DataCutterSource::processBye()
{
	//cout<<"recd bye from server!"<<endl;
}

void DataCutterSource::processError()
{
  //QString errorMessage("Error communicating with datacutter server.\nReason: ");
  //setError(QString(errorMessage + (char*)perr->getDescription()) + "\nPlease Reconnect to server", true);

	QString errorMessage("Error communicating with datacutter server.\n");
	setError(errorMessage, true);
}

	
void DataCutterSource::recvMessage()
{
	uchar *pbuffer=buffer;
	int nbytesleft=BUFSIZE;
	int nbytesread;

	freeAll();

#ifdef DEBUG2
	cout<<"client waiting for message !"<<endl;
#endif

	while(nbytesleft>0)
		{
		int test = sockfd->bytesAvailable();
		while (sockfd->bytesAvailable()==0);
		nbytesread=sockfd->readBlock((char*)pbuffer,nbytesleft);
		//if (nbytesread<=0) return;
		pbuffer+=nbytesread;
		nbytesleft-=nbytesread;
		}


	pmess=new Message(buffer);

#ifdef DEBUG2
	cout<<"Client recd message of type: "<<(int) pmess->getMType()<<endl;
	Message::displayMessage(buffer,5);
#endif

	switch(pmess->getMType())
		{
		case HELLOACK: phma=new HelloMessageAck(buffer);break;
		case DSDETAILSACK: pgetdsda=new GetDSDetailsMessageAck(buffer); break;
		case GETDSACK: 
			pgetdsa=new GetDSMessageAck(buffer); break;
		case ERRORMASK: perr=new ErrorMessage(buffer); break;
		case BYE: pbye=new ByeMessage(buffer); break;
		default: break;
		}
	return;
}

void DataCutterSource::sendMessage()
{
	int bytesleft=BUFSIZE;
	int byteswritten=0;

#ifdef DEBUG2
	cout<<"Client writing message:"<<endl;
	Message::displayMessage(buffer,10);
#endif

	char *pbuffer=(char *)buffer;

	while(bytesleft>0)
		{
		byteswritten=sockfd->writeBlock(pbuffer,bytesleft);
		bytesleft-=byteswritten;
		pbuffer+=byteswritten;
		}
}

	
void DataCutterSource::freeAll()
{
	if (pmess!=NULL) delete pmess;
	if (phm!=NULL) delete phm;
	if (phma!=NULL) delete phma;
	if (pgetdsd!=NULL) delete pgetdsd;
	if (pgetdsda!=NULL) delete pgetdsda;
	if (pgetds!=NULL) delete pgetds;
	if (pgetdsa!=NULL) delete pgetdsa;
	if (perr!=NULL) delete perr;
	if (pbye!=NULL) delete pbye;

	pmess=NULL;
	phm=NULL;
	phma=NULL;
	pgetdsd=NULL;
	pgetdsda=NULL;
	pgetds=NULL;
	pgetdsa=NULL;
	perr=NULL;
	pbye=NULL;
}

void DataCutterSource::resetError()
{
	// do nothing, reseting the error is not allowed since all errors are fatal
}

void DataCutterSource::fillData(char* data, uint xMin, uint yMin, uint zMin,
		uint xMax, uint yMax, uint zMax,
		uint xDim, uint yDim, uint zDim, uint variable, uint timeStep)
{
	if (error()) return;
	qDebug("xDim: %d, yDim: %d, zDim: %d", xDim, yDim, zDim);
	setSpec(new DSDetails(*currentSpec));
	unsigned int min[] = {xMin, yMin, zMin};
	unsigned int max[] = {xMax, yMax, zMax};
	unsigned int dim[] = {xDim, yDim, zDim};
	spec->setMin(min);
	spec->setMax(max);
	spec->setDim(dim);


	QTime t;
	t.start();

	requestGetDS();
	recvMessage();
	if (perr!=NULL) {
		processError();
	}
	else {
		processGetDSAck(data, xDim*yDim*zDim);
		qDebug("Time to download data: %d ", t.elapsed());
	}
}

void DataCutterSource::fillThumbnail(char* data, 
		unsigned int xDim, unsigned int yDim, unsigned int zDim, uint variable, uint timeStep)
{
	if (error()) return;
	setSpec(new DSDetails(*currentSpec));
	unsigned int min[3];
	unsigned int max[3];
	spec->getMin(min);
	spec->getMax(max);
	unsigned int dim[] = {xDim, yDim, zDim};
	spec->setDim(dim);

	requestGetDS();
	recvMessage();
	if (perr!=NULL)
		processError();
	else
		processGetDSAck(data, xDim*yDim*zDim);
}

unsigned int DataCutterSource::getVariableType(unsigned int variable) const
{
	// return garbage for now
	// (this should be interpreted as the Char datatype; from the look of it,
				// this is correct)
	return 0;
}


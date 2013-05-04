/******************************************************************************
				Copyright   

This code is developed within the Computational Visualization Center at The 
University of Texas at Austin.

This code has been made available to you under the auspices of a Lesser General 
Public License (LGPL) (http://www.ices.utexas.edu/cvc/software/license.html) 
and terms that you have agreed to.

Upon accepting the LGPL, we request you agree to acknowledge the use of use of 
the code that results in any published work, including scientific papers, 
films, and videotapes by citing the following references:

C. Bajaj, Z. Yu, M. Auer
Volumetric Feature Extraction and Visualization of Tomographic Molecular Imaging
Journal of Structural Biology, Volume 144, Issues 1-2, October 2003, Pages 
132-143.

If you desire to use this code for a profit venture, or if you do not wish to 
accept LGPL, but desire usage of this code, please contact Chandrajit Bajaj 
(bajaj@ices.utexas.edu) at the Computational Visualization Center at The 
University of Texas at Austin for a different license.
******************************************************************************/

#include <dataCutterClient/message.h>
#include<string.h>
//#include<iostream>

//using namespace std;

GetDSMessageAck::GetDSMessageAck():Message(GETDSACK,LITTLEENDIAN)
	{
	name=NULL;
	DataType=SCALARINT;
	NBytes=0;
	Data=NULL;
	}

GetDSMessageAck::GetDSMessageAck(uchar *buffer)
	{
	name=NULL;
	DataType=SCALARINT;
	NBytes=0;
	deSerialize(buffer);
	}

void GetDSMessageAck::setName(uchar *lname)
	{
	if (name!=NULL) delete [] name;
	if (lname==NULL)
		{
		name=NULL;
		}
	else    {
		name=new uchar[strlen((char *)lname)];
		strcpy((char*)name,(char*)lname);
		}
	return;
	}

uchar* GetDSMessageAck::getName() {return name;}

void GetDSMessageAck::setData(uchar *lData) { Data=lData; }
uchar* GetDSMessageAck::getData() { return Data; }

void GetDSMessageAck::setDataType(uchar type) { DataType=type; }
uchar GetDSMessageAck::getDataType() { return DataType; }

void GetDSMessageAck::setNBytes(uint nb) { NBytes=nb; }
uint GetDSMessageAck::getNBytes() { return NBytes; }

uchar* GetDSMessageAck::serialize(uchar *buffer)
	{
	uchar *lbuffer=Message::serialize(buffer);

	memcpy(lbuffer,name,strlen((char*)name) + 1);
	lbuffer += strlen((char*)name) + 1;

	memcpy(lbuffer,&DataType,sizeof(DataType));
	lbuffer += sizeof(DataType);

	memcpy(lbuffer,&NBytes,sizeof(NBytes));
	lbuffer += sizeof(NBytes);

	return lbuffer;

	}

uchar* GetDSMessageAck::deSerialize(uchar *buffer)
	{
	uchar *lbuffer=Message::deSerialize(buffer);
	//cout<<"\n lbuffer ="<<lbuffer<<" length = "<<(char *)lbuffer - cptr<<endl;

	if (name!=NULL) delete [] name;
	name=new uchar[strlen((char*)lbuffer)+1];

	memcpy(name,lbuffer,strlen((char*)lbuffer)+1);
	//cout<<"Name = "<<name<<endl;

	lbuffer += strlen((char*)lbuffer) + 1;

	memcpy(&DataType,lbuffer,sizeof(DataType));
	lbuffer += sizeof(DataType);

	memcpy(&NBytes,lbuffer,sizeof(NBytes));
	lbuffer += sizeof(NBytes);

	return lbuffer;
	}

GetDSMessageAck::~GetDSMessageAck()
	{
	if (name!=NULL) delete [] name;
	}

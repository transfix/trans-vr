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

#include <dataCutterClient/message.h>
#include<stdio.h>
#include<string.h>
//#include <iostream>

//using namespace std;

ErrorMessage::ErrorMessage():Message(ERRORMASK,LITTLEENDIAN)
	{
	originalMType=HELLO;
	description=NULL;
	}

ErrorMessage::ErrorMessage(uchar *buffer) {
	originalMType=HELLO;
	description=NULL;
	deSerialize(buffer); }

void ErrorMessage::setDescription(uchar *data)
	{
	if (description!=NULL) delete [] description;
	if (data==NULL)
		{
		description=NULL;
		}
	else    {
		description=new uchar[strlen((char *)data)];
		strcpy((char*)description,(char*)data);
		}
	return;
	}

uchar* ErrorMessage::getDescription() { return description; }

void ErrorMessage::setOriginalMType(uchar m) {originalMType=m;}

uchar ErrorMessage::getOriginalMType() { return originalMType; }

uchar* ErrorMessage::serialize(uchar *buffer)
	{
	uchar *lbuffer=Message::serialize(buffer);

	memcpy(lbuffer,&originalMType,sizeof(originalMType));
	lbuffer += sizeof(originalMType);

	memcpy(lbuffer,description,strlen((char*)description) + 1);
	lbuffer += strlen((char*)description) + 2;

	return lbuffer;
	}

uchar* ErrorMessage::deSerialize(uchar *buffer)
	{
	uchar *lbuffer=Message::deSerialize(buffer);
	//cout<<"\n lbuffer ="<<lbuffer<<" length = "<<(char *)lbuffer - cptr<<endl;

	memcpy(&originalMType,lbuffer,sizeof(originalMType));
	//cout<<"\nOriginal MType = "<<(int)originalMType<<endl;
	
	lbuffer += sizeof(originalMType);

	if (description!=NULL) delete [] description;
	description=new uchar[strlen((char*)lbuffer)+2];

	//cout<<"\nAllocated mem"<<endl;

	memcpy(description,lbuffer,strlen((char*)lbuffer)+1);
	//cout<<"Description = "<<description<<endl;

	lbuffer += strlen((char*)lbuffer) + 2;


	return lbuffer;

	}

ErrorMessage::~ErrorMessage()
	{
	if (description!=NULL)
		delete description;
	}

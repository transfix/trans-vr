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

GetDSMessage::GetDSMessage():Message(GETDS,LITTLEENDIAN)
	{
	name=NULL;
	spec=NULL;
	}

GetDSMessage::GetDSMessage(uchar *buffer)
	{
	name=NULL;
	spec=NULL;
	deSerialize(buffer);

	}

void GetDSMessage::setName(uchar *lname)
	{
	if (name!=NULL) delete [] name;
	if (lname==NULL)
		{
		name=NULL;
		}
	else    {
		name=new uchar[strlen((char *)lname) + 1];
		strcpy((char*)name,(char*)lname);
		}
	return;
	}

uchar* GetDSMessage::getName()
	{
	return name;
	}

void GetDSMessage::setSpec(DSDetails *det)
	{
	spec=det;
	return;
	}

DSDetails* GetDSMessage::getSpec()
	{
	return spec;
	}

uchar* GetDSMessage::serialize(uchar *buffer)
	{
	uchar *lbuffer=Message::serialize(buffer);

	memcpy(lbuffer,name,strlen((char*)name) + 1);
	lbuffer += strlen((char*)name) + 1 + 1;

	lbuffer=spec->serialize(lbuffer);

	return lbuffer;
	}

uchar* GetDSMessage::deSerialize(uchar *buffer)
	{
	uchar *lbuffer=Message::deSerialize(buffer);
	//cout<<"\n lbuffer ="<<lbuffer<<" length = "<<(char *)lbuffer - cptr<<endl;

	if (name!=NULL) delete [] name;
	name=new uchar[strlen((char*)lbuffer)+1];


	memcpy(name,lbuffer,strlen((char*)lbuffer)+1);
	//cout<<"Name = "<<name<<endl;

	lbuffer += strlen((char*)lbuffer) + 2;

	if (spec!=NULL)
		delete spec;
	spec=new DSDetails;

	lbuffer=spec->deSerialize(lbuffer);

	return lbuffer;
	}

GetDSMessage::~GetDSMessage()
	{
	if (name!=NULL) delete [] name;
	if (spec!=NULL) delete spec;
	}

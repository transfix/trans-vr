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

GetDSDetailsMessage::GetDSDetailsMessage():Message(DSDETAILS,LITTLEENDIAN) {name=NULL;}

GetDSDetailsMessage::GetDSDetailsMessage(uchar *buffer) {name=NULL; deSerialize(buffer); }

void GetDSDetailsMessage::setName(uchar *lname) {
						  if (name!=NULL) delete name;
						  name=new uchar[strlen((char*) lname) + 1];
						  strcpy((char*)name,(char*)lname);
						  }

uchar* GetDSDetailsMessage::getName() { return name; }

uchar* GetDSDetailsMessage::serialize(uchar *buffer)
	{
	uchar *lbuffer=Message::serialize(buffer);

	memcpy((char*)lbuffer,(char*)name,strlen((char *)name)+1);

	lbuffer+=strlen((char*)name) + 2;
	return lbuffer;
	}

uchar* GetDSDetailsMessage::deSerialize(uchar *buffer)
	{
	int len;
	uchar *lbuffer=Message::deSerialize(buffer);

	len=strlen((char*)lbuffer);

	if (name!=NULL) delete name;

	name=new uchar[len+1];

	strcpy((char*)name,(char*)lbuffer);

	lbuffer+=strlen((char*)name) + 2;
	return lbuffer;
	}

GetDSDetailsMessage::~GetDSDetailsMessage()
	{
	if (name!=NULL) delete name;
	}

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
#include <stdio.h>
//#include <iostream.h>

//using namespace std;

Message::Message()
	{
	//cout<<"Called Message()!"<<endl;
	//cout.flush();

	setMType(HELLO);
	setBOrder(LITTLEENDIAN);
	}

Message::Message(uchar lM, uchar lBO)
	{
	setMType(lM);
	setBOrder(lBO);
	}

Message::Message(uchar *buffer)
	{
	//cout<<"Called Message(buffer)!"<<endl;
	//cout.flush();

	deSerialize(buffer);
	}

uchar Message::getMessageType(uchar *buffer)
	{
	return *(buffer);
	}

uchar* Message::skipHeader(uchar *buffer)
	{
	return buffer + 2;
	}

void Message::displayMessage(uchar *buffer,int len)
	{
	int i;
	for (i=0;i<len;i++)
		{
		printf("\n%02x %c",buffer[i],buffer[i]);
		}
	}

void Message::setMType(uchar lM) {MTYPE=lM;}
void Message::setBOrder(uchar lBO) {BYTEORDER=lBO;}

uchar Message::getMType() {return MTYPE;}
uchar Message::getBOrder() {return BYTEORDER;}


uchar* Message::deSerialize(uchar *buffer)
	{
	setMType(buffer[0]);
	setBOrder(buffer[1]);
	//cout<<"deserialised basic message!"<<endl;
	return Message::skipHeader(buffer);
	}

uchar* Message::serialize(uchar *buffer)
	{
	buffer[0]=getMType();
	buffer[1]=getBOrder();
	return Message::skipHeader(buffer);
	}


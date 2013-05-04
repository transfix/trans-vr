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
#include<stdio.h>
#include<string.h>
//#include<iostream>

//using namespace std;

HelloMessageAck::HelloMessageAck(int lmax):Message(HELLOACK,LITTLEENDIAN)
	{
	maxsets=lmax;
	name=new uchar*[lmax];
	description=new uchar*[lmax];
	nsets=0;
	curr=0;
	}

HelloMessageAck::HelloMessageAck(uchar *buffer)
	{
	maxsets=10;
	name=new uchar*[maxsets];
	description=new uchar*[maxsets];
	nsets=0;
	curr=0;
	deSerialize(buffer);
	}

void HelloMessageAck::addSet(uchar *lname, uchar *ldesc)
	{

	if (nsets + 1>maxsets) return;
	name[nsets] = new uchar[strlen((char *)lname)+1];
	strcpy((char *) name[nsets],(char *)lname);
	description[nsets] = new uchar[strlen((char *)ldesc)+1];
	strcpy((char *)description[nsets],(char *)ldesc);
	nsets++;

	}

void HelloMessageAck::getNext(uchar *lname, uchar *ldesc)
	{
	if (curr + 1 > maxsets)
		{
		lname[0]='\0';
		ldesc[0]='\0';
		return;
		}

	strcpy((char *)lname,(char *)name[curr]);
	strcpy((char *)ldesc,(char *)description[curr]);
	curr++;
	return;
	}

void HelloMessageAck::reset()
	{
	curr=0;
	return;
	}

int HelloMessageAck::getMaxsets() {return maxsets;}
int HelloMessageAck::getNsets() {return nsets;}


uchar* HelloMessageAck::serialize(uchar *buffer)
	{
	uchar *lbuffer;
	uchar stemp[200];
	int i;

	lbuffer=Message::serialize(buffer);

	memcpy(lbuffer,&nsets,sizeof(nsets));

	lbuffer += sizeof(nsets);

	for (i=0;i<nsets;i++)
		{
		sprintf((char *)stemp,"%s:%s:",(char *)name[i],(char *)description[i]);
		memcpy(lbuffer,stemp,strlen((char *)stemp) + 1);
		lbuffer += strlen((char *)stemp);
		}
	return lbuffer;
	}

uchar* HelloMessageAck::deSerialize(uchar *buffer)
	{
	uchar ntemp[50];
	uchar dtemp[300];
	uchar *cptr;
	int n;

	int i;

	uchar *lbuffer=Message::deSerialize(buffer);

	memcpy(&n,lbuffer,sizeof(n));

	lbuffer += sizeof(n);


	for (i=0;i<n;i++)
		{

		cptr=(uchar *)strstr((char *)lbuffer,":");

		memcpy(ntemp,lbuffer,cptr - lbuffer);
		ntemp[cptr-lbuffer]='\0';

		lbuffer=cptr + 1;

		cptr=(uchar *)strstr((char *)lbuffer,":");

		memcpy(dtemp,lbuffer,cptr - lbuffer);
		dtemp[cptr-lbuffer]='\0';

		lbuffer=cptr + 1;

		//cout<<ntemp<<","<<dtemp<<endl;
		//cout<<strlen((char*)ntemp)<<","<<strlen((char*)dtemp)<<endl;
		addSet(ntemp,dtemp);
		}
	return lbuffer;
	}

HelloMessageAck::~HelloMessageAck()
	{
	int i;

	for (i=0;i<nsets;i++)
		{
		delete [] name[i];
		delete [] description[i];
		}

	delete [] name;
	delete [] description;
	}

/*
  Copyright 2002-2003 The University of Texas at Austin
  
    Authors: Anthony Thane <thanea@ices.utexas.edu>
             Vinay Siddavanahalli <skvinay@cs.utexas.edu>
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
  along with Volume Rover; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <ColorTable/IsocontourMap.h>
#include <qfile.h>

#define COLORTABLE_ISOCONTOURMAP_POSITION_AND_RGB

int IsocontourMap::IsocontourMapNode::ms_NextID = 0;


IsocontourMap::IsocontourMap()
: m_Size(0)
{
	m_pHead = 0;
	//AddNode( 0.5 );
	//AddNode( 0.6 );
}

IsocontourMap::IsocontourMap(const IsocontourMap& copy) : m_Size(copy.m_Size)
{
	if (copy.m_pHead) {
		m_pHead = new IsocontourMapNode(*(copy.m_pHead));
	}
	else {
		m_pHead = 0;
	}
}

IsocontourMap::~IsocontourMap()
{
	delete m_pHead;
}

IsocontourMap::IsocontourMapNode::IsocontourMapNode(double fPosition,
								double fR, double fG, double fB, 
								IsocontourMap::IsocontourMapNode* Next) 
										: m_fPosition(fPosition),
										m_pNext(Next), m_ID(getNextID()), m_R(fR),m_G(fG),m_B(fB)	
{
}

IsocontourMap::IsocontourMapNode::IsocontourMapNode(const IsocontourMap::IsocontourMapNode& copy)
: m_fPosition(copy.m_fPosition), m_ID(copy.m_ID),
	m_R(copy.m_R), m_G(copy.m_G), m_B(copy.m_B)
{
	if (copy.m_pNext)
		m_pNext = new IsocontourMapNode(*(copy.m_pNext));
	else 
		m_pNext = 0;
}

IsocontourMap::IsocontourMapNode::~IsocontourMapNode()
{
	delete m_pNext;
}



int IsocontourMap::AddNode(double fPosition, double fR, double fG, double fB)
{
	if (fPosition>=0.0 && fPosition<=1.0) {
		IsocontourMapNode* newNode;
		newNode = new IsocontourMapNode(fPosition, fR,fG,fB, m_pHead);
		m_pHead = newNode;
		m_Size++;

		return newNode->m_ID; //m_Size-1;
	}
	return -1;
}

void IsocontourMap::DeleteIthNode(int index)
{
	IsocontourMapNode* pNode = m_pHead;
	IsocontourMapNode* toDelete;
	
	if( index == 0 )
	{
		m_pHead = m_pHead->m_pNext;
		pNode->m_pNext = 0;
		delete pNode;
		m_Size--;
		return;
	}

	int i = 1;
	while (pNode && i<index) {
		i++;
		pNode = pNode->m_pNext;
	}
	if (pNode && pNode->m_pNext) {
		toDelete = pNode->m_pNext;
		pNode->m_pNext = toDelete->m_pNext;
		toDelete->m_pNext = 0;
		delete toDelete;
		m_Size--;
	}
}

void IsocontourMap::MoveIthNode(int index, double fPosition)
{
	if (index<m_Size) {
		IsocontourMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (pNode && i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			if (fPosition > 1.0) {
				fPosition = 1.0;
			}
			else if (fPosition < 0.0) {
				fPosition = 0.0;
			}
			pNode->m_fPosition = fPosition;
		}
	}
}

int IsocontourMap::GetIDofIthNode(int index) const
{
	if (index<m_Size && index>=0) {
		IsocontourMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (pNode && i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			return pNode->m_ID;
		}
	}

	// failed to get id
	return -1;
}

void IsocontourMap::ChangeColor(int index, double fR, double fG, double fB)
{
	if (index<m_Size && index>=0) {
		IsocontourMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (pNode && i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			pNode->m_R = fR;
			pNode->m_G = fG;
			pNode->m_B = fB;
		}
	}
}

double IsocontourMap::GetRed(int index) const
{
	if (index<m_Size && index>=0) {
		IsocontourMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (pNode && i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			return pNode->m_R;
		}
	}

	// failed to get color
	return 0.0;
}

double IsocontourMap::GetGreen(int index) const
{
	if (index<m_Size && index>=0) {
		IsocontourMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (pNode && i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			return pNode->m_G;
		}
	}

	// failed to get color
	return 0.0;
}

double IsocontourMap::GetBlue(int index) const
{
	if (index<m_Size && index>=0) {
		IsocontourMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (pNode && i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			return pNode->m_B;
		}
	}

	// failed to get color
	return 0.0;
}

int IsocontourMap::GetSize() const
{
	return m_Size;
}

double IsocontourMap::GetPosition(int id) const
{
	IsocontourMapNode* pNode;
	pNode = m_pHead;
	int i = 0;
	while (pNode && pNode->m_ID != id) {
		i++;
		pNode = pNode->m_pNext;
	}
	if (pNode) {
		return pNode->m_fPosition;
	}
	else {
		return 0.0;
	}
}

double IsocontourMap::GetPositionOfIthNode(int index) const
{
	if (index < m_Size && index >= 0) {
		IsocontourMapNode* pNode;
		pNode = m_pHead;
		int i = 0;
		while (i<index) {
			i++;
			pNode = pNode->m_pNext;
		}
		if (pNode) {
			return pNode->m_fPosition;
		}
		else {
			return 0.0;
		}
	}
	else {
		return 0.0;
	}
}

/********************

IsocontourMap
No of nodes
< no of nodes >
Position
< Position A1 >
< Position A2 >
< Position A3 >
< Position A4 >
...
< Position AN >

*********************/
void IsocontourMap::saveMap( QTextStream& stream )
{
	IsocontourMapNode* isocontourMapNode;

	stream << "IsocontourMap\n";
	stream << "Number of nodes\n";
	stream << m_Size << "\n";

#ifndef COLORTABLE_ISOCONTOURMAP_POSITION_AND_RGB
	stream << "Position\n";


	isocontourMapNode = m_pHead;

	for( int i=0;i<m_Size; i++ )
	{
		stream << isocontourMapNode->m_fPosition << "\n";
		isocontourMapNode = isocontourMapNode->m_pNext;
	}
#else
	stream << "Position and RGB\n";


	isocontourMapNode = m_pHead;

	for( int i=0;i<m_Size; i++ )
	{
		stream << isocontourMapNode->m_fPosition << " " <<isocontourMapNode->m_R << " " <<isocontourMapNode->m_G << " " <<isocontourMapNode->m_B << "\n";
		isocontourMapNode = isocontourMapNode->m_pNext;
	}
#endif
}

void IsocontourMap::loadMap( QTextStream& stream )
{
	IsocontourMapNode* pHead, *node;
	double position,red,green,blue;
	int size;
	QString junk;



	stream.skipWhiteSpace();
	junk = stream.readLine(); // IsocontourMap

	stream.skipWhiteSpace();
	junk = stream.readLine(); // Number of nodes
	stream >> size;

	if (size != 0) {

		stream.skipWhiteSpace();
		junk = stream.readLine(); // Position
		bool with_rgb = false;
		if(junk == "Position")
		  {
		    stream >> position;
		    pHead = new IsocontourMapNode(position, 0.5,0.5,0.5, 0);
		    with_rgb = false;
		  }
		else if(junk == "Position and RGB")
		  {
		    stream >> position >> red >> green >> blue;
		    pHead = new IsocontourMapNode(position, red, green, blue, 0);
		    with_rgb = true;
		  }

		node = pHead;

		for( int i=1;i<size; i++ ) {
		  if(!with_rgb)
		    {
		      stream >> position;
		      node->m_pNext = new IsocontourMapNode(position, 0.5,0.5,0.5, 0);
		    }
		  else
		    {
			stream >> position >> red >> green >> blue;
			node->m_pNext = new IsocontourMapNode(position, red,green,blue, 0);
		    }
		  node = node->m_pNext;

		}


		// if successful, replace lists
		delete m_pHead;
		m_pHead = pHead;
		m_Size = size;
	}
	else {
		delete m_pHead;
		m_pHead = 0;
		m_Size = 0;
	}

}


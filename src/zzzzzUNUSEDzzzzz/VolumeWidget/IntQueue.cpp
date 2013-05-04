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
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

// IntQueue.cpp: implementation of the IntQueue class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeWidget/IntQueue.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

QueueNode::QueueNode(QueueNode* next, int index)
{
	m_Prev = 0;
	m_Next = next;
	if (next) next->m_Prev = this;
	m_Index = index;
}

QueueNode::~QueueNode()
{
	delete m_Next;
	m_Next = 0;

}

Queue::Queue()
{
	m_Head = 0;
	m_Tail = 0;
}

Queue::Queue(const Queue& copy)
{
	m_Head = 0;
	m_Tail = 0;
	QueueNode* p = m_Tail;
	while (p) {
		enQueue(p->m_Index);
		p = p->m_Prev;
	}
}

Queue::~Queue()
{
	delete m_Head;
	m_Head = 0;
	m_Tail = 0;
}


Queue& Queue::enQueue(int index)
{
	m_Head = new QueueNode(m_Head, index);
	if (m_Tail==0) {
		m_Tail = m_Head;
	}
	return *this;
}

int Queue::deQueue()
{
	if (!isEmpty()) {
		unsigned int index = m_Tail->m_Index;
		QueueNode* temp = m_Tail;
		
		m_Tail = m_Tail->m_Prev;
		if (m_Tail) {
			m_Tail->m_Next = 0;
		}
		else {
			m_Head = 0;
		}

		delete temp;

		return index;
	}
	else {
		return -1;
	}
}

void Queue::clearQueue()
{
	delete m_Head;
	m_Head = 0;
	m_Tail = 0;
}

bool Queue::isEmpty() const
{
	return (m_Head==0);
}


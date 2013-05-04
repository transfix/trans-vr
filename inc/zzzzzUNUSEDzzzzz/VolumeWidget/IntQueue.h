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

// IntQueue.h: interface for the IntQueue class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_INTQUEUE_H__5BCDDDD5_C74D_413B_A351_382DAF0EA82A__INCLUDED_)
#define AFX_INTQUEUE_H__5BCDDDD5_C74D_413B_A351_382DAF0EA82A__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

class QueueNode {
public:
	QueueNode(QueueNode* next, int index);
	QueueNode(const QueueNode& copy);
	QueueNode& operator=(const QueueNode& copy);

	~QueueNode();
	QueueNode* m_Prev;
	QueueNode* m_Next;
	int m_Index;
};


class Queue {
public:
	Queue();
	Queue(const Queue& copy);
	~Queue();

	Queue& enQueue(int index);
	int deQueue();
	void clearQueue();
	bool isEmpty() const;
	QueueNode* m_Head;
	QueueNode* m_Tail;
};

#endif // !defined(AFX_INTQUEUE_H__5BCDDDD5_C74D_413B_A351_382DAF0EA82A__INCLUDED_)

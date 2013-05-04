/*
  Copyright 2000-2003 The University of Texas at Austin

	Authors: Xiaoyu Zhang 2000-2002 <xiaoyu@ices.utexas.edu>
					 John Wiggins 2003 <prok@cs.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of iotree.

  iotree is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  iotree is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
#ifndef VERTEX_H
#define VERTEX_H

//#define ARITH_ENCODE
//#define ZP_CODEC

const unsigned char POSX = 0x01;
const unsigned char POSY = 0x02;
const unsigned char POSZ = 0x04;
const unsigned char NEGX = 0x08;
const unsigned char NEGY = 0x10;
const unsigned char NEGZ = 0x20;

static unsigned char edgedir[6] = 
{POSX, POSY, POSZ, NEGX, NEGY, NEGZ};

template <class T>
class Slice;

/**
 */
template <class T>
class Vertex {
private:
	T      val;
	unsigned char code;

public:
	Vertex(){val = 0; code = 0;}
	Vertex(const T& x):val(x){code = 0;}
	Vertex(const Vertex<T>& vtx):val(vtx.val){code = vtx.code;}
	~Vertex(){}

	Vertex<T>& operator= (const Vertex<T>& vtx); 
	void setValue(const T& x, unsigned char c = 0);
	void setBit(unsigned char);
	bool isUsed() const;
	const T& getValue() { return val;}

	// reset the code bits
	void reset() {
		code = 0;
	}

	friend class Slice<T>;
};

template<class T>
inline void Vertex<T>::setBit(unsigned char direct)
{
	code |= direct;
} 

template<class T>
inline bool Vertex<T>::isUsed() const
{
	return(code != 0x0);
}

template<class T>
Vertex<T>& Vertex<T>::operator=(const Vertex<T> &vx)
{
	if (this != &vx) {
		val = vx.val;
		code = vx.code;
	}
	return *this;
}

template<class T>
inline void Vertex<T>::setValue(const T& x, unsigned char c)
{
	val = x;
	code = c;
}

#endif


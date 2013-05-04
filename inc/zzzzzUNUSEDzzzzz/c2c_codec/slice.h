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
#ifndef C2C_SLICE_H
#define C2C_SLICE_H

#include <vector>
#include <iostream>
#include <stdio.h>
#include <c2c_codec/vertex.h>
#include <arithlib/bitbuffer.h>
#include <arithlib/encode.h>
#include <c2c_codec/diskio.h>
#include <libdjvupp/ByteStream.h>
#include <c2c_codec/util.h>

//#define SQUARE_ZAG

using namespace std;

template <class T> class CompCon;
template <class T> class SingleConStep;

template <class T>
class Slice {
protected:
    int  dim[2];       // x and y dimension of slice
    Vertex<T> *verts;  // vertices of the slice
		unsigned char *colors; // RGB vertex colors
    int  nv;           // number of vertices used
		DiskIO* out;
		ByteStream* bstream;
		int out_flag;	   // out_flag: 0: write to a diskio, 1: write to a stream
		int ID;
		static int sID;

public:
    ///
    Slice(FILE *input, FILE *output, int dx, int dy);

    Slice(T* data, unsigned char *red, unsigned char *green,
									 unsigned char *blue, int *_dim);

    Slice(const Slice<T>& sl);

    ~Slice();

		/// write the slice in encoded format to a DiskIO
    void writeOut(DiskIO*, float);

		/// write the slice in encoded format to a ByteStream
		void writeOut(ByteStream* stream, float val);

    /// get the byte size of slice
    int size() {
			if (colors)
				return (sizeof(T)+3) * dim[0] * dim[1];
			else
				return sizeof(T) * dim[0] * dim[1];
		}
    
		// mark the vertex(ix, iy) using code
    void setBit(int ix, int iy, u_char code);

    // reset codes of the vertex array
    void reset();

    friend class CompCon<T>;
    friend class BitBuffer;
    friend class SingleConStep<T>;

    /**
     *accessors
	 */
	//@{
    int width() {return dim[0];}
    int height(){return dim[1];}
    T*  getData() {
        T* data = new T[dim[0]*dim[1]];
        for (int i = 0; i < dim[1]; i++)
            for (int j = 0; j < dim[0]; j++) {
                data[i*dim[0]+j] = verts[i*dim[0]+j].val;
            }
        return data;
    }
	//@}
	
    BIT *diffBits(const Slice<T>& sl);

protected:
    void runLength();		 // runlength encode indices of used vertices
    void arithIndex();		 // arithmetical encode indices of used vertices
    void encodeVerts(float); // encode function values of used vertices
		void writeColors(); // write the unencoded colors of used vertices
};

template <class T>
int Slice<T>::sID = 0;

template <class T>
Slice<T>::Slice(T* data, unsigned char *red, unsigned char *green,
												 unsigned char *blue, int *_dim)
{
    dim[0] = _dim[0]; dim[1] = _dim[1];
    nv = 0;
		ID = sID++;
    verts = new Vertex<T>[dim[0]*dim[1]];
    for (int i = 0; i < dim[0]*dim[1]; i++) {
        verts[i].setValue(data[i]);
    }

		if (red && green && blue) {
			//printf("max color index = %d\n", dim[0]*dim[1]*3 - 1);
			colors = new unsigned char [dim[0]*dim[1]*3];

    	for (int i = 0; i < dim[0]*dim[1]; i++) {
					colors[i*3+0] = red[i];
					colors[i*3+1] = green[i];
					colors[i*3+2] = blue[i];
  	  }
		}
		else
			colors = 0; // no color

}

template<class T>
Slice<T>::Slice(const Slice<T>& sl)
{
    dim[0] = sl.dim[0]; dim[1] = sl.dim[1];
    nv = sl.nv;
		ID = sl.ID;
    verts = new Vertex<T>[dim[0]*dim[1]];
    for (int i = 0; i < dim[0]*dim[1]; i++) {
        verts[i] = sl.verts[i];
        //verts[i].setValue(sl.verts[i].val, sl.verts[i].code);
    }

		if (sl.colors) {
			colors = new unsigned char [dim[0]*dim[1]*3];
			for (int i=0; i < dim[0]*dim[1]*3; i++) {
				colors[i] = sl.colors[i];
			}
		}
		else
			colors = 0; // no color
}

// destructor
template <class T>
inline Slice<T>::~Slice()
{
    if (verts != NULL) delete[] verts;
		if (colors != NULL) delete [] colors;
}

template <class T>
void Slice<T>::writeOut(DiskIO* _out, float isoval)
{
	out = _out;
	out_flag = 0;
    out->put(&nv, 1);   
#ifdef _DEBUG
    printf("[%d]# of used vertices = %d\n", ID, nv);
#endif

#ifdef ARITH_ENCODE
    arithIndex();
	//runLength();
#else
    runLength();
#endif

    encodeVerts(isoval);

		if (colors)
			writeColors();
}

template <class T>
void Slice<T>::writeOut(ByteStream* _stream, float isoval)
{
	bstream = _stream;
	out_flag = 1;
	bstream->write32(nv);

#ifdef _DEBUG
    printf("[%d]# of used vertices = %d\n", ID, nv);
#endif

#ifdef ARITH_ENCODE
    arithIndex();
#else
    runLength();
#endif

    encodeVerts(isoval);

		if (colors)
			writeColors();
}

template <class T>
void Slice<T>::writeColors()
{
	if (nv == 0) return;

	//printf("writeColors(): nv = %d\n", nv);

	unsigned char *cols = new unsigned char [nv*3];
	int count = 0;
	for (int j=0; j < dim[1]; j++) {
		for (int i=0; i < dim[0]; i++) {
			int n = i + j*dim[0];
			if (verts[n].isUsed()) {
				//printf("count = %d, n = %d\n", count, n);
				cols[count*3+0] = colors[n*3+0];
				cols[count*3+1] = colors[n*3+1];
				cols[count*3+2] = colors[n*3+2];
				count++;
			}
		}
	}

	//printf("writing\n");

	switch (out_flag) {
		case 0:
			out->put(cols, nv*3);
			break;
		case 1:
			bstream->writall(cols, nv*3);
			break;
	}

	delete[] cols;
}

template <class T>
void Slice<T>::encodeVerts(float isoval)
{
    if (nv == 0) return;

    T *pval = new T[nv];
    int count = 0;
    for (int j = 0; j < dim[1]; j++) {
        for (int i = 0; i < dim[0]; i++) {
            int n = i + j*dim[0];
            if (verts[n].isUsed()) {
                pval[count++] = verts[n].val;
            }
        }
    }
    BitBuffer * valbuf = encode_vals(pval, nv, isoval);
	switch(out_flag) {
	case 0:
		writeBitBuffer(valbuf, out);
		break;
	case 1:
		writeBitBuffer(valbuf, bstream);
		break;
	}
    
#ifdef _DEBUG
    printf("slice function data uses %d compressed bytes\n", valbuf->getNumBytes());
#endif
    delete[] pval;
    delete   valbuf;

}

template <class T>
void Slice<T>::runLength()
{
    if (nv == 0) return;

	BitBuffer *indbuf = new BitBuffer();
	int len = 0;
	BIT curbit = 0;     // current bit
	BIT *bits = new BIT[dim[0]*dim[1]];

#ifdef SQUARE_ZAG
    // arrange bits in continuous order
    for (int j = 0; j < dim[1]; j++) {
        for (int i = 0; i < dim[0]; i++) {
            int n = i + j*dim[0];
            if (j%2 == 0) {
                if (verts[n].isUsed()) bits[n] = 1;
                else bits[n] = 0;
            } else {
                int m = dim[0]*(j+1) - i - 1;
                if (verts[m].isUsed()) bits[n] = 1;
                else bits[n] = 0;
            }
        }
    }
#elif defined(TRIANGLE_ZAG)
    int i = 0, j = 0, n = 0;
    int dir = 0;            // ZAG direction
    while (i < dim[0] && j < dim[1]) {
        int m = i + j*dim[0];
        if (verts[m].isUsed()) bits[n++] = 1;
        else bits[n++] = 0;
        if (j == 0 && dir == 0) {
            if (i < dim[0]-1)i++;
            else j++;
            dir = (dir + 1)%2;
            continue;
        }

        if (i == 0 && dir == 1) {
            if (j < dim[1]-1) j++;
            else i++;
            dir = (dir + 1)%2;
            continue;
        }

        if ( i == dim[0]-1 && dir == 0) {
            j++;
            dir = (dir+1)%2;
            continue;
        }

        if (j== dim[1]-1 && dir == 1) {
            i++;
            dir = (dir+1)%2;
            continue;
        }

        if (dir == 0) {
            i++; j--;
            continue;
        }

        if (dir == 1) {
            i--; j++;
            continue;
        }
    }
#else
    for (int j = 0; j < dim[1]; j++) {
        for (int i = 0; i < dim[0]; i++) {
            int n = i + j*dim[0];
            bits[n] =  (verts[n].isUsed())? 1:0;
        }
    }
#endif

	for(int j = 0; j < dim[1]; j++) {
		for(int i = 0; i < dim[0]; i++) {
			int n = i + j*dim[0];
			if(bits[n] == curbit) len++;
			else {
				indbuf->Encode_Positive(len);
				curbit = bits[n];
				len = 1;
			}
		}
	}
	if(len > 0) indbuf->Encode_Positive(len);
	/*
#ifdef ZP_CODEC
    indbuf->zp_encode();
#else
    indbuf->arith_encode();
#endif  //  ZP_CODEC
	*/
	switch(out_flag) {
	case 0:
		writeBitBuffer(indbuf, out);
		break;
	case 1:
		writeBitBuffer(indbuf, bstream);
		break;
	}
#ifdef _DEBUG
    printf("run length slice bitmap uses %d compressed bytes\n", indbuf->getNumBytes());
#endif
    delete indbuf;
    delete[] bits;

	/***	
    int nstrip = 0;
    unsigned char len = 0;
    vector<unsigned char> vec;
    u_char startx, starty;   // starting cell of a strip
    bool stripOn = false;

    for (int j = 0; j < dim[1]; j++) {
        for (int i = 0; i < dim[0]; i++) {
            int n = i + j*dim[0];
            if (!stripOn && verts[n].isUsed()) { // a strip starts
                startx = i; starty = j;
                nstrip++;
                len = 1;
                stripOn = true;
            } else if (stripOn && verts[n].isUsed() && len < 255) { // strip continues
                len ++;
                // here is an assumption that len <= 256, will fix it later
            } else if (stripOn && (!verts[n].isUsed() || len == 255)) { // strip ends
                vec.push_back(startx);
                vec.push_back(starty);
                vec.push_back(len);
                stripOn = false;
            } else {
                // do nothing
            }
        }
    }

    if (stripOn && len > 0) {  // add the last strip if existing
        vec.push_back(startx);
        vec.push_back(starty);
        vec.push_back(len);
    }

    printf("num of vertex strips = %d, compressed bytes = %d\n", nstrip, nstrip*3);
    assert(vec.size() == nstrip*3);
    //fwrite(&nstrip, sizeof(int), 1, out_file);
    for (int i = 0; i < nstrip*3; i++) {
		switch(out_flag) {
		case 0:
			out->put(&vec[i], 1);
			break;
		case 1:
			bstream->write8(vec[i]);
			break;
		}
    } 
	*/ 
}

template <class T>
void Slice<T>::arithIndex() 
{
    if (nv == 0) return;    // no index needs to be encoded

    BIT *bits = new BIT[dim[0]*dim[1]];
#ifdef SQUARE_ZAG

    // arrange bits in continuous order
    for (int j = 0; j < dim[1]; j++) {
        for (int i = 0; i < dim[0]; i++) {
            int n = i + j*dim[0];
            if (j%2 == 0) {
                if (verts[n].isUsed()) bits[n] = 1;
                else bits[n] = 0;
            } else {
                int m = dim[0]*(j+1) - i - 1;
                if (verts[m].isUsed()) bits[n] = 1;
                else bits[n] = 0;
            }
        }
    }
#elif defined(TRIANGLE_ZAG)
    int i = 0, j = 0, n = 0;
    int dir = 0;            // ZAG direction
    while (i < dim[0] && j < dim[1]) {
        int m = i + j*dim[0];
        if (verts[m].isUsed()) bits[n++] = 1;
        else bits[n++] = 0;
        if (j == 0 && dir == 0) {
            if (i < dim[0]-1)i++;
            else j++;
            dir = (dir + 1)%2;
            continue;
        }

        if (i == 0 && dir == 1) {
            if (j < dim[1]-1) j++;
            else i++;
            dir = (dir + 1)%2;
            continue;
        }

        if ( i == dim[0]-1 && dir == 0) {
            j++;
            dir = (dir+1)%2;
            continue;
        }

        if (j== dim[1]-1 && dir == 1) {
            i++;
            dir = (dir+1)%2;
            continue;
        }

        if (dir == 0) {
            i++; j--;
            continue;
        }

        if (dir == 1) {
            i--; j++;
            continue;
        }
    }
#else
    for (int j = 0; j < dim[1]; j++) {
        for (int i = 0; i < dim[0]; i++) {
            int n = i + j*dim[0];
            bits[n] =  (verts[n].isUsed())? 1:0;
        }
    }
#endif

    BitBuffer *indbuf = new BitBuffer();
    indbuf->put_bits(dim[0]*dim[1], bits);
#ifdef ZP_CODEC
    indbuf->zp_encode();
#else
    indbuf->arith_encode();
#endif  //  ZP_CODEC
	switch(out_flag) {
	case 0:
		writeBitBuffer(indbuf, out);
		break;
	case 1:
		writeBitBuffer(indbuf, bstream);
		break;
	}

#ifdef _DEBUG
    printf("slice bitmap uses %d compressed bytes\n", indbuf->getNumBytes());
#endif
    delete indbuf;
    delete[] bits;
}

template <class T>
inline void Slice<T>::setBit(int ix, int iy, u_char code)
{
    int n = ix + iy*dim[0];
    if (!verts[n].isUsed()) nv++;  // increment the used vertex # by 1
    verts[n].setBit(code);
}

template <class T>
inline void Slice<T>::reset() 
{
    nv = 0;
    for (int i = 0; i < dim[0]*dim[1]; i++) {
        verts[i].reset();
    }
}

template<class T>
BIT *Slice<T>::diffBits(const Slice<T>& sl)
{
    if (dim[0] != sl.dim[0] || dim[1] != sl.dim[1]) {
        fprintf(stderr, "slice size doesn't match\n");
        return NULL;
    }

    BIT *bits = new BIT[dim[0]*dim[1]];
    for (int j = 0; j < dim[1]; j++)
        for (int i = 0; i < dim[0]; i++) {
            int n = i + j*dim[0];
            bits[n] = (sl.verts[n].isUsed() == verts[n].isUsed())? 0:1;
        }
    return bits;
}

#endif


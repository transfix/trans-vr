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
/******
// bufferedio.h: interface for the BufferedIO class.
// It inherits from the DiskIO interface. We use the DiskIO as an 
// interface to support possible different I/O characteristics
//
******/

#if !defined(AFX_BUFFEREDIO_H__52F9DCBF_BBD5_406E_9344_E46BEF997CF2__INCLUDED_)
#define AFX_BUFFEREDIO_H__52F9DCBF_BBD5_406E_9344_E46BEF997CF2__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <stdio.h>
#include <c2c_codec/diskio.h>
#include <c2c_codec/config.h>

/** Buffered I/O 
*/
class BufferedIO  : virtual public DiskIO
{
public:
	///
	BufferedIO(const char* fname, DiskIO::Type mode = DiskIO::READ, int buf_size = NDB);
	///
	BufferedIO(FILE* fp, DiskIO::Type mode = DiskIO::READ, int buf_size = NDB);
	///
	virtual ~BufferedIO();

	/**@name Get Functions 
	 * overloaded get functions for various data type
	 */
	//@{
	///
	virtual int get(char* data, int size);
	///
	virtual int get(unsigned char* data, int size);
	///
	virtual int get(short* data, int size);
	///
	virtual int get(unsigned short* data, int size);
	///
	virtual int get(int* data, int size);
	///
	virtual int get(long* data, int size);
	///
	virtual int get(int64* data, int size);
	///
	virtual int get(float* data, int size);
	///
	virtual int get(double* data, int size);
	//@}

	/**@name Put Functions 
	 * overloaded put functions for various data type
	 */
	//@{
	///
	virtual void put(const char* data, int size);
	///
	virtual void put(const unsigned char* data, int size);
	///
	virtual void put(const short* data, int size);
	///
	virtual void put(const unsigned short* data, int size);
	///
	virtual void put(const int* data, int size);
	///
	virtual void put(const long* data, int size);
	///
	virtual void put(const int64* data, int size);
	///
	virtual void put(const float* data, int size);
	///
	virtual void put(const double* data, int size);
	//@}

	/// open the disk IO for read or write
	virtual bool open();

	/// open another file
	virtual bool reopen(const char *fname);

	/**	reopen the current file
	 * 	write files will be opened in append mode
	 */ 
	virtual bool reopen();

	/// close the disk IO
	virtual bool close(bool fill = true);

	/** In the case of write mode : write out everything in buffer 
	    and clear the buffer. If fill = true, file will be appended until next
	    disk block boundary.
	    In the case of read mode : move pointer to the beginnign of next disk block
	    This is useful when reader want to skip stuff in current disk block.
	 */
	virtual void flush(bool fill = true);

	/// inherited seek function
	virtual bool seek(long offset, DiskIO::SeekMode mode);

	/// change file access type
	virtual bool setMode(DiskIO::Type type);

	/// end of file has been reached
	bool eof() {return (m_eof && m_bufptr == m_bufsize);}

private:
	/**@name Primitive I/O operations
	 * raw read and write functions 
	 */
	//@{
	/** get #n# bytes from buffer, go to disk if necessary
	@param size unit record size
	@param n	number of records
	@return number of records actually read
	*/
	int getraw(void *data, int size, int n);
	/// put #n# bytes to buffer, flust if necessary
	void putraw(const void *data, int size);
	//@}

	void init();			// initialize the buffer

	//enum {NDB = 20};       // enum hack of const variable
	const static int NDB;
	char* m_fname;
	FILE* m_fp;
	//char m_buffer[DBSIZE * NDB];
	int m_ndb;
	char* m_buffer;
	int  m_total;			// the whole size of the total buffer = NDB*DBSIZE
	int  m_bufsize;         // current size of buffer
	int  m_bufptr;          // current position of the buffer
	bool m_eof;
};

#endif // !defined(AFX_BUFFEREDIO_H__52F9DCBF_BBD5_406E_9344_E46BEF997CF2__INCLUDED_)


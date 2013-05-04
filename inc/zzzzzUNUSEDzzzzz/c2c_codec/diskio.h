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
 *
 * DiskIO : It's a interface that encapsulates the details of disk file access.
 * It hides the small and big endian problem. You access the stream by using 
 * the overloaded get and put methods.
 * 
 * (c) 2000 Xiaoyu Zhang
 *
 ******/
#ifndef XYZ_DISKIO_H
#define XYZ_DISKIO_H

#include <sys/types.h>

#include <c2c_codec/config.h>
/** DiskIO abstract base interface
	It hides the small and big endian problem. Real disk access classes are
	subclass of this interface
	*/
class DiskIO {
public:
	enum Type { READ = 0, WRITE };

	enum SeekMode {FROM_HERE = 0, FROM_START, FROM_END};

	///
	DiskIO(Type mode = READ) { m_mode = mode; }

	virtual ~DiskIO() {}

	/// accessors
	virtual int get(char* data, int size) = 0;

	///
	virtual int get(unsigned char* data, int size) = 0;

	///
	virtual int get(short* data, int size) = 0;

	///
	virtual int get(unsigned short* data, int size) = 0;

	///
	virtual int get(int* data, int size) = 0;

	///
	virtual int get(long* data, int size) = 0;

	///
	virtual int get(int64* data, int size) = 0;

	///
	virtual int get(float* data, int size) = 0;

	///
	virtual int get(double* data, int size) = 0;

	/// mutators
	virtual void put(const char* data, int size) = 0;

	///
	virtual void put(const unsigned char* data, int size) = 0;

	///
	virtual void put(const short* data, int size) = 0;

	///
	virtual void put(const unsigned short* data, int size) = 0;

	///
	virtual void put(const int* data, int size) = 0;

	///
	virtual void put(const long* data, int size) = 0;

	///
	virtual void put(const int64* data, int size) = 0;

	///
	virtual void put(const float* data, int size) = 0;

	///
	virtual void put(const double* data, int size) = 0;

	///
	Type mode() const { return m_mode;}

	/// change the file access type 
	virtual bool setMode(DiskIO::Type type) {return false;}

	/// open the disk IO for read or write
	virtual bool open() = 0;

	/// open another file, but cannot change type
	virtual bool reopen(const char* fname) = 0;

 	/**	reopen the current file
	 * 	write files will be opened in append mode
	 */ 
	virtual bool reopen() { return false; }

	/** close the disk IO
		@param fill fill the rest of current disk block
	*/
	virtual bool close(bool fill = true) = 0;

	/** seek
		@return true successful
		@return false failed
	*/
	virtual bool seek(long offset, SeekMode mode) = 0;

	/** write out everything in buffer 
		and possiblly move pointer to next available disk block
		@param fill fill the rest of current disk block
	*/
	virtual void flush(bool fill = true) = 0;

protected:
	Type m_mode;
};

#endif


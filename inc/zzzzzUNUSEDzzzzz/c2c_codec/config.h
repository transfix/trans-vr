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
#if !defined(XYZ_CONFIG_H)
#define XYZ_CONFIG_H

// disk block size
#define DBSIZE 4096

// maximum disk blocks difference in one seek
#define MAX_DB_NUM 300000

// maximum filename length
#define MAX_FN_LEN 512

// replication factor
#define REP_FAC 6

//#define _LITTLE_ENDIAN 1

#ifdef WIN32
	typedef __int64 int64;
#else
	typedef long long int64;
#endif		// WIN32

// Disk block pointer(index)
typedef int64 Pointer;

#ifdef WIN32
#define STRDUP(x) _strdup(x)
#else
#define STRDUP(x) strdup(x)
#endif		//WIN32

#endif


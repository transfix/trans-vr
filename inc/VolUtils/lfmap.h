/*
  Copyright 2005-2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolUtils.

  VolUtils is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolUtils is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: lfmap.h 4742 2011-10-21 22:09:44Z transfix $ */

/*
 *  lfmap.h
 *  
 *
 *  Created by Jose  Rivera on 1/19/06.
 *  Copyright 2006 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef LFMAP_H
#define LFMAP_H

#if defined(__WIN32__) || defined(__WIN64__)
#define __WINDOWS__
#endif

/* requires at least Windows 2000 for GlobalMemoryStatusEx() */
#ifdef __WINDOWS__
#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0500
#define UNICODE
#include <windows.h>
#endif

//#ifdef __WINDOWS__
#ifdef WIN32
typedef unsigned __int64 lfmap_uint64_t;
#else
typedef unsigned long long lfmap_uint64_t;
#endif

typedef unsigned char * lfmap_ptr_t;
typedef int lfmap_mode_t;

typedef struct
{
  lfmap_uint64_t  offset;                /* initial offset from start of file */
  lfmap_uint64_t  size;                  /* number of bytes of the file that this lfmap object 
					    will use starting from offset */
  
  lfmap_ptr_t     ptr;                   /* pointer to the mapped file in memory */
  lfmap_uint64_t  map_offset;            /* offset of the current map (page aligned) */
  lfmap_uint64_t  map_size;              /* number of bytes of the file mapped to memory
					    (i.e. ptr points to a chunk at least map_size in size) */

  lfmap_uint64_t  mem_usage;             /* the maximum amount of memory in bytes to use for mapping the file to memory */
#ifndef __WINDOWS__
  int             fd;                    /* file descriptor for the mapped file */
  int             prot;                  /* mmap protection */
  int             flags;                 /* mmap flags */
#else
  HANDLE          hFile;                 /* handle for the mapped file */
  HANDLE          hFileMappingObject;    /* handle to the mapping object */
  DWORD           flProtect;             /* file mapping protection */
  DWORD           dwDesiredAccess;       /* MapViewOfFile access */
#endif
} lfmap_t;

#define LFMAP_READ       ((lfmap_mode_t)0x00000001)   /* pages may be read */
#define LFMAP_WRITE      ((lfmap_mode_t)0x00000002)   /* pages may be written */
#define LFMAP_EXEC       ((lfmap_mode_t)0x00000004)   /* pages may be executed */
#define LFMAP_PRIVATE    ((lfmap_mode_t)0x00000008)   /* modifications are private */
#define LFMAP_SHARED     ((lfmap_mode_t)0x00000010)   /* modifications are shared */

#define LFMAP_DEFAULT_USAGE 5.0

#ifdef __cplusplus
extern "C" {
#endif

  /*
    Create an lfmap object.  Returns an lfmap object allocated by malloc(), or NULL if an error occurred.
    filename - file name of object to be mapped
    len - number of bytes of the file to map
    mode - Or'ed values of LFMAP_READ, LFMAP_WRITE, LFMAP_EXEC, 
    and exactly one of either LFMAP_PRIVATE or LFMAP_SHARED
    offset - map file starting at offset.
    mem_usage - the maximum amount of memory to use for this map (in percentage of main memory)
  */
  lfmap_t *lfmap_create(const char *filename, lfmap_uint64_t len, lfmap_mode_t mode, lfmap_uint64_t offset, double mem_usage);

  /*
    Destroy an lfmap object.  Will free any used resources including freeing the 'lfmap' pointer.
    lfmap - the lfmap object to destroy.
  */
  void lfmap_destroy(lfmap_t *lfmap);
 
  /*
    Returns a pointer to the location 'index' in the 'lfmap' object, or NULL if an error occurred.
    lfmap - the lfmap object to index.
    index - the index into the lfmap object Must be less than lfmap->size.
    min_size - ensure that the length of the block of data pointed to
    by the returned pointer is at least 'min_size' in length.  Must be greater than 0.
  */
  lfmap_ptr_t lfmap_ptr(lfmap_t *lfmap, lfmap_uint64_t index, lfmap_uint64_t min_size);

  /*
    Returns TRUE if the whole chunk of data that the lfmap object
    refers to is mapped to memory.  If this is the case, lfmap_ptr does not need to swap,
    so the macro lfmap_ptr_fast may be used in its place.
  */
#define lfmap_completely_mapped(lfmap) ((lfmap)->size==(lfmap)->map_size)

#define lfmap_ptr_fast(lfmap, index, min_size) ((lfmap_ptr_t)((lfmap)->ptr+((lfmap)->offset+(index)-(lfmap)->map_offset)))

#ifdef __cplusplus
};
#endif

#endif

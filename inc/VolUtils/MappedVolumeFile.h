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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

/* $Id: MappedVolumeFile.h 4742 2011-10-21 22:09:44Z transfix $ */

#ifndef MAPPEDVOLUMEFILE_H
#define MAPPEDVOLUMEFILE_H

#ifdef __WINDOWS__
#define _WIN32_WINNT 0x0500
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <VolUtils/lfmap.h>

#define SWAP_64(a)                                                           \
  {                                                                          \
    unsigned char tmp[8];                                                    \
    unsigned char *ch;                                                       \
    ch = (unsigned char *)(a);                                               \
    tmp[0] = ch[0];                                                          \
    tmp[1] = ch[1];                                                          \
    tmp[2] = ch[2];                                                          \
    tmp[3] = ch[3];                                                          \
    tmp[4] = ch[4];                                                          \
    tmp[5] = ch[5];                                                          \
    tmp[6] = ch[6];                                                          \
    tmp[7] = ch[7];                                                          \
    ch[0] = tmp[7];                                                          \
    ch[1] = tmp[6];                                                          \
    ch[2] = tmp[5];                                                          \
    ch[3] = tmp[4];                                                          \
    ch[4] = tmp[3];                                                          \
    ch[5] = tmp[2];                                                          \
    ch[6] = tmp[1];                                                          \
    ch[7] = tmp[0];                                                          \
  }
#define SWAP_32(a)                                                           \
  {                                                                          \
    unsigned char tmp[4];                                                    \
    unsigned char *ch;                                                       \
    ch = (unsigned char *)(a);                                               \
    tmp[0] = ch[0];                                                          \
    tmp[1] = ch[1];                                                          \
    tmp[2] = ch[2];                                                          \
    tmp[3] = ch[3];                                                          \
    ch[0] = tmp[3];                                                          \
    ch[1] = tmp[2];                                                          \
    ch[2] = tmp[1];                                                          \
    ch[3] = tmp[0];                                                          \
  }
#define SWAP_16(a)                                                           \
  {                                                                          \
    unsigned char d;                                                         \
    unsigned char *ch;                                                       \
    ch = (unsigned char *)(a);                                               \
    d = ch[0];                                                               \
    ch[0] = ch[1];                                                           \
    ch[1] = d;                                                               \
  }
static inline int big_endian() {
  long one = 1;
  return !(*((char *)(&one)));
}

class MappedVolumeFile;

class Variable {
public:
  enum VariableType { UCHAR, USHORT, UINT, FLOAT, DOUBLE };

  Variable(MappedVolumeFile *vf, lfmap_uint64_t var, const char *name,
           VariableType vartype, bool doswap);
  ~Variable();

  double get(lfmap_uint64_t i, lfmap_uint64_t j,
             lfmap_uint64_t
                 k); /* returns double for all types since it's the biggest */
  void
  get(lfmap_uint64_t x, lfmap_uint64_t y, lfmap_uint64_t z,
      lfmap_uint64_t dimx, lfmap_uint64_t dimy, lfmap_uint64_t dimz,
      void *buf); /* fills the memory pointed to by buf with volume data
                     according to the subvolume specified. The byte order for
                     the volume data is also swapped if m_Swap is true. Make
                     sure buf points to enough memory. The type of the data in
                     buf is the same as m_VariableType */
  unsigned char getMapped(lfmap_uint64_t i, lfmap_uint64_t j,
                          lfmap_uint64_t k);
  void getMapped(lfmap_uint64_t x, lfmap_uint64_t y, lfmap_uint64_t z,
                 lfmap_uint64_t dimx, lfmap_uint64_t dimy,
                 lfmap_uint64_t dimz, unsigned char *buf);
  inline const char *name() { return m_Name; }
  inline lfmap_uint64_t size() { return m_Size; }
  inline const VariableType getType() const { return m_VariableType; }

  double m_Min, m_Max;

private:
  double fastget(lfmap_uint64_t i, lfmap_uint64_t j, lfmap_uint64_t k);
  double slowget(lfmap_uint64_t i, lfmap_uint64_t j, lfmap_uint64_t k);

  MappedVolumeFile *m_MappedVolumeFile;
  char m_Name[256];
  lfmap_uint64_t m_Volume; /* start index of variable */
  VariableType m_VariableType;
  bool m_Swap; /* if this is true, then swap the output value of get() */
  lfmap_uint64_t m_Size;
};

class MappedVolumeFile {
public:
  MappedVolumeFile(
      const char *filename,
      bool calc_minmax =
          true); /* if calc_minmax is false, getMapped() in variables wont
                    work correctly until min and max is calculated */
  MappedVolumeFile(const char *filename, double mem_usage,
                   bool calc_minmax = true);
  virtual ~MappedVolumeFile();

  /* returns false if this volume file could not be read, or is closed */
  bool isValid() { return m_Valid; }

  inline Variable *get(unsigned int var, unsigned int time) const {
    return m_Variables[var + m_NumVariables * time];
  }

  inline unsigned int numVariables() { return m_NumVariables; }
  inline unsigned int numTimesteps() { return m_NumTimesteps; }
  inline lfmap_t *LFMappedVolumeFile() { return m_LFMappedVolumeFile; }
  inline unsigned int XDim() const { return m_XDim; }
  inline unsigned int YDim() const { return m_YDim; }
  inline unsigned int ZDim() const { return m_ZDim; }
  inline double XSpan() const { return m_XSpan; }
  inline double YSpan() const { return m_YSpan; }
  inline double ZSpan() const { return m_ZSpan; }
  inline double TSpan() const { return m_TSpan; }
  inline const char *filename() const { return m_Filename; }

protected:
  bool open(double mem_usage);
  void close();
  virtual bool
  readHeader() = 0; /* set up everything specific to a volume format here */

  char m_Filename[4096];

  bool m_Valid;
  unsigned int m_NumVariables; /* number of variables */
  Variable *
      *m_Variables; /* timestep and variable list. see get() for indexing  */

  unsigned int m_NumTimesteps; /* number of timesteps */

  lfmap_uint64_t m_XDim;
  lfmap_uint64_t m_YDim;
  lfmap_uint64_t m_ZDim;

  double m_XSpan;
  double m_YSpan;
  double m_ZSpan;
  double m_TSpan;

  lfmap_uint64_t m_Filesize;

  /*
     Use the following to use lfmap instead of directly mapping the volume
     file in the case that a volume file would use up most of the
     application's address space. (i.e. 32-bit systems loading a large file)
   */
  lfmap_t *m_LFMappedVolumeFile;

  bool m_CalcMinMax;
};

#endif

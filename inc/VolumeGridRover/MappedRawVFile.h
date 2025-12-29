/*
  Copyright 2005 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeGridRover.

  VolumeGridRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeGridRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#ifndef MAPPEDRAWVFILE_H
#define MAPPEDRAWVFILE_H

#include <VolumeGridRover/MappedVolumeFile.h>

class MappedRawVFile : public MappedVolumeFile {
public:
  MappedRawVFile(const char *filename, bool calc_minmax = true,
                 bool forceLoad = false);
  MappedRawVFile(const char *filename, double mem_usage,
                 bool calc_minmax = true, bool forceLoad = false);
  ~MappedRawVFile();

protected:
  bool readHeader();

private:
  bool m_ForceLoad; /* if this is true, load the file even if the header error
                       checking fails */
};

#endif

/*
  Copyright 2002-2003 The University of Texas at Austin

        Authors: Anthony Thane <thanea@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeLibrary.

  VolumeLibrary is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeLibrary is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#ifndef __VOLUME_LIBRARY_CG_PROGRAMS_H__
#define __VOLUME_LIBRARY_CG_PROGRAMS_H__

#include <Cg/cg.h>
#include <Cg/cgGL.h>

#ifdef __VOLUME_LIBRARY_CG_PROGRAMS_CPP__
#define CGGlobal
#else
#define CGGlobal extern
#endif

#define MAX_CG_PROGRAM 64

#define DEF_CG_RGBA 0x01
#define DEF_CG_TRANS 0x02

CGGlobal CGcontext _contextCg;
CGGlobal CGprofile _vertexProfileCg;
CGGlobal CGprofile _fragmentProfileCg;
CGGlobal CGprogram _vertexProgramCg[MAX_CG_PROGRAM];
CGGlobal CGprogram _fragmentProgramCgShaded[MAX_CG_PROGRAM];
CGGlobal CGprogram _fragmentProgramCgUnshaded[MAX_CG_PROGRAM];
CGGlobal int _vertexProgramCgCounter;
CGGlobal int _fragmentProgramCgCounter;

class CG_Programs {
public:
  ~CG_Programs();

protected:
  CG_Programs();

  static void handleCgError();

  bool initCG(int renderType);
  bool ChooseProfiles();
  bool LoadCgPrograms(int renderType);

  static bool _cgErrorFlag;

  int _cgVertexProgramId;
  int _cgFragmentProgramId;
};

#endif

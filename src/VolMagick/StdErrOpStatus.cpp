/*
  Copyright 2007-2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolMagick.

  VolMagick is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolMagick is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: StdErrOpStatus.cpp 4742 2011-10-21 22:09:44Z transfix $ */

#include <VolMagick/StdErrOpStatus.h>

#include <cstdio>

using namespace std;

namespace VolMagick
{
  void StdErrOpStatus::start(const VolMagick::Voxels *, 
			     Operation, 
			     VolMagick::uint64 numSteps) const
  {
    _numSteps = numSteps;
  }

  void StdErrOpStatus::step(const VolMagick::Voxels *,
			    Operation op, 
			    VolMagick::uint64 curStep) const
  {
    double percent = _numSteps > 1 ?
      (((float)curStep)/((float)((int)(_numSteps-1))))*100.0 :
      100.0;
    fprintf(stderr,"%s: %5.2f %%\r",opStrings[op],percent);
  }

  void StdErrOpStatus::end(const VolMagick::Voxels *,
			   Operation) const
  {
    fprintf(stderr,"\n");
  }
}

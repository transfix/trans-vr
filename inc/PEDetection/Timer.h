/*
  Copyright 2006 The University of Texas at Austin

        Authors: Sangmin Park <smpark@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of PEDetection.

  PEDetection is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  PEDetection is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#ifndef FILE_TIMER_H
#define FILE_TIMER_H

#include <iostream>
#include <time.h>

class Timer {
protected:
  // struct timeval		tv;
  // double				dBegin, dEnd;
  time_t dBegin, dEnd;

public:
  Timer() {
    dBegin = 0;
    dEnd = 0;
  }

  ~Timer() {}

  void Start() {
    // gettimeofday(&tv, NULL);
    // dBegin = (double) tv.tv_sec + tv.tv_usec / 1000000.0;
    dBegin = time(NULL);
  }

  void End(char *Comment) {
    // gettimeofday(&tv, NULL);
    // dEnd = (double) tv.tv_sec + tv.tv_usec / 1000000.0;
    dEnd = time(NULL);

    std::cout << Comment << " : Total CPU time = " << dEnd - dBegin << " Sec";
    std::cout << " (= " << float(dEnd - dBegin) / 60.0 << " Min) ";
    if ((float(dEnd - dBegin) / 60.0) > 60.0)
      std::cout << "(= " << float(dEnd - dBegin) / 3600.0 << " Hours)";
    std::cout << std::endl << std::endl;
    std::cout.flush();
  }
};

#endif

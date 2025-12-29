#ifndef __GENERATEMESH_H
#define __GENERATEMESH_H

#include <string>

void generateMesh(string manifestFile, float isoratio, float tolerance,
                  float volthresh, int meshStart, int meshEnd,
                  string outPref);

#endif

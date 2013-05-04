#ifndef __GENERATERAWV_H
#define __GENERATERAWV_H

#include<string>

#include<VolMagick/VolMagick.h>


void generateRawVFromVolume(string inpvol, int dimension, float edgel, string prefix);
//void generateRawVFromVolume(VolMagick::Volume& inpvol, int dimension, float edgel, string prefix);
void generateRawVFromMesh(string inpVol, int meshStart, int meshEnd, string meshPrefix, int dimension, float edgeLength, string outpref);


#endif

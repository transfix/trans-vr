/*
  Copyright 2006-2007 The University of Texas at Austin

        Authoris: Dr. Xu Guo Liang <xuguo@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of HLevelSet.

  HLevelSet is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  HLevelSet is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

// HLevelSet.h: interface for the HLevelSet class.
//
//////////////////////////////////////////////////////////////////////

#ifndef _HLEVELSET_H_

class SimpleVolumeData;
namespace PDBParser {
class GroupOfAtoms;
class Atom;
}; // namespace PDBParser

#include <VolMagick/VolMagick.h>
#include <boost/tuple/tuple.hpp>
#include <vector>

namespace HLevelSetNS {

class HLevelSet {
public:
  HLevelSet();
  HLevelSet(float el, int e, int md);
  virtual ~HLevelSet();

  //		SimpleVolumeData* getHigherOrderLevelSetSurface(
  //PDBParser::GroupOfAtoms* molecule, unsigned int* dim );
  bool computeFunction_ajrkN(float *func_h, float *func_phi,
                             unsigned int *dim, float *minExtent,
                             float *maxExtent);
  boost::tuple<bool, VolMagick::Volume> getHigherOrderLevelSetSurface_Xu_Li(
      const std::vector<float> vertex_Positions, unsigned int *dim,
      float edgelength, int end, int max_dim);
  boost::tuple<bool, VolMagick::Volume> getHigherOrderLevelSetSurface_Xu_Li_N(
      const std::vector<float> vertex_Positions, unsigned int *dim,
      VolMagick::BoundingBox &bb,
      /*VolMagick::Volume &coeffVol,*/ float &isovalue);

  boost::tuple<bool, VolMagick::Volume>
  getHigherOrderLevelSetSurface_Gauss_Xu_Li(const std::vector<float> Vertices,
                                            const std::vector<float> Radius,
                                            unsigned int *dim);
  boost::tuple<bool, VolMagick::Volume>
  getHigherOrderLevelSetSurface_sdf(float edgelength, int end);
  boost::tuple<bool, VolMagick::Volume>
  getHigherOrderLevelSetSurface(const std::vector<float> Vertices,
                                const std::vector<float> Radius,
                                unsigned int *dim);
  VolMagick::Volume computeHLSCoefficients(VolMagick::Volume &vol);

protected:
  //	bool getAtomListAndExtent( PDBParser::GroupOfAtoms* molecule,
  //std::vector<PDBParser::Atom*> &atomList, float* minExtent, float*
  //maxExtent );
  //      bool computeFunction( std::vector<PDBParser::Atom*> atomList, float*
  //      data, unsigned int* dim, float* minExtent, float* maxExtent );
  bool computeFunction(const std::vector<float> Vertices,
                       const std::vector<float> Radius, float *func_phi,
                       unsigned int *dim, float *minExtent, float *maxExtent);

  //	bool computeFunction_Xu( std::vector<PDBParser::Atom*> atomList,
  //float* data, unsigned int* dim, float* minExtent, float* maxExtent );
  bool computeFunction_Zhang(std::vector<float> vertex_Positions,
                             float *funcvalue, unsigned int *dim,
                             float *minExt, float *maxExt, float edgelength,
                             int end);
  bool computeFunction_Zhang_N(std::vector<float> vertex_Positions,
                               float *funcvalue, unsigned int *dim,
                               float *minExt, float *maxExt, float &isovalue);
  bool computeFunction(std::vector<float> vertex_Positions, float *funcvalue,
                       unsigned int *dim, float *minExt, float *maxExt);

  bool computeFunction_Zhang_sdf(float *funcvalue, float edgelength, int end);
  bool computeFunction_Xu_Li(float *vertJPosition, int size, float *gridvalue,
                             float *funcvalue, unsigned int *dim,
                             float *minExt, float *maxExt, int *newobject,
                             int numobject, float edgelength);
  bool computeFunction_Xu_Li_N(float *vertJPosition, int size,
                               float *gridvalue, float *funcvalue,
                               unsigned int *dim, float *minExt,
                               float *maxExt, int *newobject, int numobject);

private:
  unsigned int Dim[3];
  int End;
  float *Coefficent;
  float *Funcvalue;
  float *Funcvalue_bak, *Dfang;
  float Dxyz[3], MinExt[3], MaxExt[3], Dt;
  float *boundarysign;
  int Max_dim, end;
  float edgelength;
};

}; // namespace HLevelSetNS

#endif

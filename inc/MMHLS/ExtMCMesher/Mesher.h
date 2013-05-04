/***************************************************************************
 *   Copyright (C) 2009 by Bharadwaj Subramanian   *
 *   bharadwajs@pupil.ices.utexas.edu   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#ifndef __MESHER_H
#define __MESHER_H

#include "Grid.h"
#include "Mesh.h"
#include "MCTester.h"
#include <VolMagick/VolMagick.h>
#include <vector>
#include <map>
#include <string>

using namespace std;

struct Minimizer
{
  Point p;
  vector<unsigned int> edgeIndices;
};

class Mesher
{
  private:
    Grid *grid;
    Mesh *mesh;
    map<unsigned int, vector<Minimizer> > boundaryMinimizers;
    map<unsigned int, MCCase> boundaryMCCases;
    vector<unsigned int> boundaryCells;
    
    vector<Minimizer> computeMinimizers(Cell &c,float isovalue,MCCase &mcCase);
    Minimizer getMinimizer(Cell &c,float isovalue,vector<unsigned int> edges);
    Point getMinimizerForEdge(unsigned int index, unsigned int edgeIndex);
    vector<Point> computeDualElement(unsigned int index,unsigned int edgeIndex, float isovalue,vector<Index> neighbors);

  public:
    void setGrid(Grid &g) { grid=&g; }
    void setMesh(Mesh *m) { mesh=m; }
    void generateMesh(float isovalue);
    void saveMesh(string filename,VolMagick::Volume *volume);
};

#endif

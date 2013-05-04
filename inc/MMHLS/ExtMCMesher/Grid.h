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
#ifndef __GRID_H
#define __GRID_H

#include <VolMagick/VolMagick.h>
#include <vector>

using namespace std;

struct Index
{
  unsigned int i,j,k;
};

struct Point
{
  float x,y,z;
};

struct Cell
{
  float vertexValues[8];
  Point vertexLocations[8];
  unsigned int xDim,yDim,zDim;
};   

class Grid
{
  public:
    VolMagick::Volume *volume;
    vector<VolMagick::Volume> gradients;
  public:
    /// Import the volume into the grid.
    virtual void importVolume(VolMagick::Volume &v)=0;
    /// Get the number of cells in the volume.
    virtual unsigned int numCells(void)=0;
    /// Get the dimensions of the volume (number of cells in x,y,z direction).
    virtual Index getDimensions(void)=0;
    /// Get the cell based upon a single 1-D index.
    virtual Cell getCell(unsigned int index)=0;
    /// Get the cell within which the given point p belongs
    virtual Cell getCell(Point &p)=0;
    /// Get the cell based on an i,j,k index.
    virtual bool getCell(Index &index,Cell &c)=0;
    /// Get the neighboring cell indices for a given vertex.
    virtual vector<Index> getVertexNeighbors(Index &cellIndex,unsigned int vertexIndex)=0;
    /// Get the neighboring cell indices for a given edge
    virtual vector<Index> getEdgeNeighbors(Index &cellIndex,unsigned int edgeIndex)=0;
    /// Get the neighboring cell indices for a given face.
    virtual Index getFaceNeighbor(Index &cellIndex,unsigned int faceIndex)=0;
    /// Get the neighboring cell indices for a given cell.
    virtual vector<Index> getCellNeighbors(Index &cellIndex)=0;
    /// Test if the given cell is a boundary cell based on provided isovalue.
    virtual bool isBoundaryCell(Index &index,float isovalue)=0;
    /// Local function; converts an Index to an index.
    virtual unsigned int get1DIndex(Index &index)=0;
    /// Local function; converts an index to an Index.
    virtual Index get3DIndex(unsigned int index)=0;
    /// Checks if a given cell given by an Index exists.
    virtual bool cellExists(Index &index)=0;
    /// Returns the gradient at a point.
    virtual Point getGradient(Point p)=0;
};

#endif

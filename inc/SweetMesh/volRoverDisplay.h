/***************************************************************************
 *   Copyright (C) 2010 by Jesse Sweet   *
 *   jessethesweet@gmail.com   *
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

#ifndef VOLROVERDISPLAY_H
#define VOLROVERDISPLAY_H

#include <SweetMesh/hexmesh.h>
#include <SweetMesh/meshTools.h>
#include <cvcraw_geometry/cvcgeom.h>


namespace sweetMesh{

class visualMesh : public CVCGEOM_NAMESPACE::cvcgeom_t{
public:
	visualMesh()	{}
	visualMesh(hexMesh& mesh) { meshPtr = &mesh; init_ptrs(); }
	~visualMesh()	{}

	bool 		renderAllEdges;
	bool 		renderAllSurfaceQuads;

	points_t	customPoints;		//Note: typedefs points_t,
	boundary_t	customBoundaries;	//boundary_t, normals_t, etc.
	normals_t	customNormals;		//are defined in the file
	colors_t	customColors;		//cvcraw_geometry/cvcgeom.h
	lines_t		customLines;
	triangles_t	customTriangles;
	quads_t		customQuads;

	void		refresh();
	void		clear();

protected:
	hexMesh*	meshPtr;

private:
	void 		refreshVertices();
	void 		refreshLines();
	void 		refreshQuads();
};

}

#endif

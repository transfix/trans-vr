/*
 This file is part of the VRender library.
 Copyright (C) 2005 Cyril Soler (Cyril.Soler@imag.fr)
 Version 1.0.0, released on June 27, 2005.

 http://artis.imag.fr/Members/Cyril.Soler/VRender

 VRender is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 VRender is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with VRender; if not, write to the Free Software Foundation, Inc.,
 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.
*/

/**********************************************************************

Copyright (C) 2002-2025 Gilles Debunne. All rights reserved.

This file is part of the QGLViewer library version 3.0.0.

https://gillesdebunne.github.io/libQGLViewer - contact@libqglviewer.com

This file is part of a free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program; if not, write to the Free Software Foundation,
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.

**********************************************************************/

#include <QGLViewer/VRender/Optimizer.h>
#include <QGLViewer/VRender/Primitive.h>
#include <QGLViewer/VRender/VRender.h>
#include <vector>

using namespace std;
using namespace vrender;

// Over-simplified algorithm to check wether a polygon is front-facing or not.
// Only works for convex polygons.

void BackFaceCullingOptimizer::optimize(
    std::vector<PtrPrimitive> &primitives_tab, VRenderParams &) {
  Polygone *P;
  int nb_culled = 0;

  for (size_t i = 0; i < primitives_tab.size(); ++i)
    if ((P = dynamic_cast<Polygone *>(primitives_tab[i])) != nullptr) {
      for (unsigned int j = 0; j < P->nbVertices(); ++j)
        if (((P->vertex(j + 2) - P->vertex(j + 1)) ^
             (P->vertex(j + 1) - P->vertex(j)))
                .z() > 0.0) {
          delete primitives_tab[i];
          primitives_tab[i] = nullptr;
          ++nb_culled;
          break;
        }
    }

  // Rule out gaps. This avoids testing for null primitives later.

  int j = 0;
  for (size_t k = 0; k < primitives_tab.size(); ++k)
    if (primitives_tab[k] != nullptr)
      primitives_tab[j++] = primitives_tab[k];

  primitives_tab.resize(j);
#ifdef DEBUG_BFC
  cout << "Backface culling: " << nb_culled << " polygons culled." << endl;
#endif
}

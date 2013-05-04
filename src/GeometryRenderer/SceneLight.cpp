/*
  Copyright 2012 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of libCVC.

  libCVC is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  libCVC is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: SceneLight.cpp 5746 2012-06-17 23:21:45Z transfix $ */

#include <GeometryRenderer/SceneLight.h>

namespace CVC_NAMESPACE
{
#if 0
  void SceneLight::operator()()
  {
    // arand, 7-19-2011
    // I am not sure if this code setting up the lighting options 
    // actually does anything... I just tried to copy TexMol
    glLightfv(GL_LIGHT0, GL_POSITION, position);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseColor);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specularColor);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientColor);
    
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    
    glEnable(GL_LIGHT0);
    //glEnable(GL_LIGHT1);
    glEnable(GL_NORMALIZE);
    
    //// arand: added to render both sides of the surface... 4-12-2011
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
  }
#endif
}

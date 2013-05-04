/*
  Copyright 2002-2003 The University of Texas at Austin

        Authors: Anthony Thane <thanea@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of FastContouring.

  FastContouring is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  FastContouring is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <string.h>
#include <stdexcept>
#include <FastContouring/MarchingCubesBuffers.h>
#include <FastContouring/cubes.h>

namespace FastContouring
{
  MarchingCubesBuffers::MarchingCubesBuffers()
  {
    setDefaults();
  }

  MarchingCubesBuffers::MarchingCubesBuffers(const MarchingCubesBuffers& copy)
  {
    m_AmountAllocated = copy.m_AmountAllocated;
    for(unsigned int i = 0; i < 5; i++)
      {
	m_EdgeCaches[i].reset(new unsigned int[m_AmountAllocated]);
	memcpy(m_EdgeCaches[i].get(),
	       copy.m_EdgeCaches[i].get(),
	       m_AmountAllocated*sizeof(unsigned int));
      }
    
    for(unsigned int i = 0; i < 2; i++)
      {
	m_VertClassifications[i].reset(new unsigned int[m_AmountAllocated]);
	memcpy(m_VertClassifications[i].get(),
	       copy.m_VertClassifications[i].get(),
	       m_AmountAllocated*sizeof(unsigned int));
      }
  }

  MarchingCubesBuffers::~MarchingCubesBuffers()
  {
    destroyEdgeBuffers();
  }

  MarchingCubesBuffers& MarchingCubesBuffers::operator=(const MarchingCubesBuffers& copy)
  {
    m_AmountAllocated = copy.m_AmountAllocated;
    for(unsigned int i = 0; i < 5; i++)
      {
	m_EdgeCaches[i].reset(new unsigned int[m_AmountAllocated]);
	memcpy(m_EdgeCaches[i].get(),
	       copy.m_EdgeCaches[i].get(),
	       m_AmountAllocated*sizeof(unsigned int));
      }
    
    for(unsigned int i = 0; i < 2; i++)
      {
	m_VertClassifications[i].reset(new unsigned int[m_AmountAllocated]);
	memcpy(m_VertClassifications[i].get(),
	       copy.m_VertClassifications[i].get(),
	       m_AmountAllocated*sizeof(unsigned int));
      }
    return *this;
  }

  void MarchingCubesBuffers::setDefaults()
  {
    for(EdgeCacheArray::iterator i = m_EdgeCaches.begin();
	i != m_EdgeCaches.end();
	i++)
      i->reset();

    for(VertexClassificationsArray::iterator i = m_VertClassifications.begin();
	i != m_VertClassifications.end();
	i++)
      i->reset();

    m_AmountAllocated = 0;
  }

  bool MarchingCubesBuffers::allocateEdgeBuffers(unsigned int width, unsigned int height)
  {
    // only allocate memory if our current buffer is not big enough
    if (width*height>m_AmountAllocated) {
      destroyEdgeBuffers();
      return forceAllocateEdgeBuffers(width, height);	
    }
    else {
      return true;
    }
  }

  bool MarchingCubesBuffers::forceAllocateEdgeBuffers(unsigned int width, unsigned int height)
  {
    try
      {
	m_AmountAllocated = width*height;
	for(EdgeCacheArray::iterator i = m_EdgeCaches.begin();
	    i != m_EdgeCaches.end();
	    i++)
	  i->reset(new unsigned int[m_AmountAllocated]);
	      
	for(VertexClassificationsArray::iterator i = m_VertClassifications.begin();
	    i != m_VertClassifications.end();
	    i++)
	  i->reset(new unsigned int[m_AmountAllocated]);
      }
    catch(const std::exception& e)
      {
	return false;
      }
    return true;
  }

  void MarchingCubesBuffers::destroyEdgeBuffers()
  {
    setDefaults();
  }

  void MarchingCubesBuffers::swapEdgeBuffers()
  {
    // swap the edges buffers
    boost::shared_array<unsigned int> temp;
    temp = m_EdgeCaches[XEdgesBack];
    m_EdgeCaches[XEdgesBack] = m_EdgeCaches[XEdgesFront];
    m_EdgeCaches[XEdgesFront] = temp;
  
    temp = m_EdgeCaches[YEdgesBack];
    m_EdgeCaches[YEdgesBack] = m_EdgeCaches[YEdgesFront];
    m_EdgeCaches[YEdgesFront] = temp;
  
    temp = m_VertClassifications[0];
    m_VertClassifications[0] = m_VertClassifications[1];
    m_VertClassifications[1] = temp;
  }

}

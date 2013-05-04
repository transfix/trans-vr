/*
  Copyright 2006 The University of Texas at Austin

        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of LBIE.

  LBIE is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  LBIE is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <stdio.h>
#include <algorithm>
#include <LBIE/quality_improve.h>

namespace LBIE
{
  void copyGeoframeToGeometry(const geoframe& input, boost::shared_ptr<Geometry>& output)
  {
    output.reset(new Geometry());

    int real_numverts = 0;

    std::vector<int> verts;
    
    for(int i = 0; i < input.numverts; i++)
      if(input.bound_sign[i])
	{
	  verts.push_back(i);
	  real_numverts++;
	}

    output->m_NumPoints = real_numverts;

    if(input.numtris > 0)
      {
	int real_numtris = 0;
	for(int i = 0; i < input.numtris; i++)
	  {
	    if(input.bound_sign[input.triangles[i][0]] &&
	       input.bound_sign[input.triangles[i][1]] &&
	       input.bound_sign[input.triangles[i][2]])
	      real_numtris++;
	  }
	output->m_NumTris = real_numtris;

	output->AllocateTris(output->m_NumTriVerts,output->m_NumTris);

	for(int i=0;i<verts.size();i++)
	  for(int j=0; j<3; j++)
	    output->m_TriVerts[i*3+j] = input.verts[verts[i]][j];
	
	int tri_counter = 0;
	for(int i=0;i<input.numtris;i++)
	  if(input.bound_sign[input.triangles[i][0]] &&
	     input.bound_sign[input.triangles[i][1]] &&
	     input.bound_sign[input.triangles[i][2]])
	    {
	      for(int j=0; j<3; j++)
		{
		  output->m_Tris[(tri_counter)*3+j] = input.triangles[i][j];
		}
	      tri_counter++;
	    }
	if(input.color.size() == input.numverts)
	  {
	    //we have color information, so lets grab it!
	    output->AllocatePointColors();
	    for(int i=0;i<verts.size();i++)
	      for(int j=0; j<3; j++)
		output->m_TriVertColors[i*3+j] = input.color[verts[i]][j];
	  }
      }

    if(input.numquads > 0)
      {
	int real_numquads = 0;
	for(int i = 0; i < input.numquads; i++)
	  {
	    if(input.bound_sign[input.quads[i][0]] &&
	       input.bound_sign[input.quads[i][1]] &&
	       input.bound_sign[input.quads[i][2]] &&
	       input.bound_sign[input.quads[i][3]])
	      real_numquads++;
	  }
	output->m_NumQuads = real_numquads;

	output->AllocateQuads(output->m_NumQuadVerts,output->m_NumQuads);

	for(int i=0;i<verts.size();i++)
	  for(int j=0; j<3; j++)
	    output->m_QuadVerts[i*3+j] = input.verts[verts[i]][j];

	int quad_counter = 0;
	for(int i=0;i<input.numquads;i++)
	  if(input.bound_sign[input.quads[i][0]] &&
	     input.bound_sign[input.quads[i][1]] &&
	     input.bound_sign[input.quads[i][2]] &&
	     input.bound_sign[input.quads[i][3]])
	    {
	      for(int j=0; j<4; j++)
		{
		  output->m_Quads[quad_counter*4+j] = input.quads[i][j];
		}
	      quad_counter++;
	    }
	if(input.color.size() == input.numverts)
	  {
	    //we have color information, so lets grab it!
	    output->AllocatePointColors();
	    for(int i=0;i<verts.size();i++)
	      for(int j=0; j<3; j++)
		output->m_PointColors[i*3+j] = input.color[verts[i]][j];
	  }
      }
  }

  void copyGeoframeToGeometry(const geoframe& input, Geometry& output)
  {
    output.ClearGeometry();

    if(input.numtris > 0)
      {
	output.m_NumTriVerts = input.numverts;
	output.m_NumTris = input.numtris;

	output.AllocateTris(output.m_NumTriVerts,output.m_NumTris);

	for(int i=0;i<input.numverts;i++)
	  for(int j=0; j<3; j++)
	    output.m_TriVerts[i*3+j] = input.verts[i][j];

	for(int i=0;i<input.numtris;i++)
	  for(int j=0; j<3; j++)
	    output.m_Tris[i*3+j] = input.triangles[i][j];

	if(input.color.size() == input.numverts)
	  {
	    //we have color information, so lets grab it!
	    output.AllocatePointColors();
	    for(int i=0;i<input.numverts;i++)
	      for(int j=0; j<3; j++)
		output.m_TriVertColors[i*3+j] = input.color[i][j];
	  }
      }
  }

  void copyGeometryToGeoframe(const boost::shared_ptr<Geometry>& input, geoframe& output)
  {
    if(input->m_NumTris > 0)
      {
	input->GetReadyToDrawSmooth(); //make sure we have normal data

	output.numverts = input->m_NumTriVerts;
	output.numtris = input->m_NumTris;

	//output.verts  = (float (*)[3])malloc(sizeof(float[3]) * output.numverts);
	//output.triangles   = (unsigned int (*)[3])malloc(sizeof(unsigned int[3]) * output.numtris);
	output.verts.resize(output.numverts);
	output.normals.resize(output.numverts);
	output.triangles.resize(output.numtris);
	for(int i=0;i<output.numverts;i++)
	  {
	    for(int j=0; j<3; j++)
	      output.verts[i][j] = input->m_TriVerts[i*3+j];
	    for(int j=0; j<3; j++)
	      output.normals[i][j] = input->m_TriVertNormals[i*3+j];
	  }
	for (int i=0;i<output.numtris;i++)
	  for(int j=0; j<3; j++)
	    output.triangles[i][j] = input->m_Tris[i*3+j];

	output.bound_sign.resize(output.numverts);
	std::fill(output.bound_sign.begin(),
		  output.bound_sign.end(),
		  1);
	output.bound_tri.resize(output.numtris);
	std::fill(output.bound_tri.begin(),
		  output.bound_tri.end(),
		  1);

	if(input->m_TriVertColors)
	  {
	    output.color.resize(output.numverts);
	    for(int i=0;i<output.numverts;i++)
	      for(int j=0; j<3; j++)
		output.color[i][j] = input->m_TriVertColors[i*3+j];
	  }
      }
  }

  void copyGeometryToGeoframe(Geometry& input, geoframe& output)
  {
    if(input.m_NumTris > 0)
      {
	input.GetReadyToDrawSmooth(); //make sure we have normal data

	printf("input.m_NumTris: %d\n",input.m_NumTris);

	output.reset();
	output.numverts = input.m_NumTriVerts;
	output.numtris = input.m_NumTris;

	//output.verts  = (float (*)[3])malloc(sizeof(float[3]) * output.numverts);
	//output.triangles   = (unsigned int (*)[3])malloc(sizeof(unsigned int[3]) * output.numtris);
	output.verts.resize(output.numverts);
	output.normals.resize(output.numverts);
	output.triangles.resize(output.numtris);
	for(int i=0;i<output.numverts;i++)
	  {
	    for(int j=0; j<3; j++)
	      output.verts[i][j] = input.m_TriVerts[i*3+j];
	    for(int j=0; j<3; j++)
	      output.normals[i][j] = input.m_TriVertNormals[i*3+j];
	  }
	for (int i=0;i<output.numtris;i++)
	  for(int j=0; j<3; j++)
	    output.triangles[i][j] = input.m_Tris[i*3+j];

	output.bound_sign.resize(output.numverts);
	std::fill(output.bound_sign.begin(),
		  output.bound_sign.end(),
		  1);
	output.bound_tri.resize(output.numtris);
	std::fill(output.bound_tri.begin(),
		  output.bound_tri.end(),
		  1);

	if(input.m_TriVertColors)
	  {
	    output.color.resize(output.numverts);
	    for(int i=0;i<output.numverts;i++)
	      for(int j=0; j<3; j++)
		output.color[i][j] = input.m_TriVertColors[i*3+j];
	  }
      }
  }

  boost::shared_ptr<Geometry> mesh(const VolMagick::Volume& vol,
				   float iso_val,
				   float iso_val_in,
				   float err_tol)
  {
    Octree oc;

    geoframe g_frame;
    boost::shared_ptr<Geometry> output/*(new Geometry())*/;
    double l_err=err_tol;
    //    double l_err = DEFAULT::ERR;

    //    iso_val_in = iso_val = -0.5001;

    oc.set_isovalue(iso_val*-1.0);
    oc.set_isovalue_in(iso_val_in*-1.0);

    printf("set volume\n");
    oc.setVolume(vol);
    oc.setMeshType(0);
    oc.setNormalType(0);

    printf("finish volume setting\n");

    g_frame.setSpan(oc.spans[0],oc.spans[1],oc.spans[2]);
    g_frame.setMin(oc.minext[0],oc.minext[1],oc.minext[2]);
    oc.collapse();
    printf("collaspse\n");
    oc.compute_qef();
    printf("computer qef\n");
    oc.traverse_qef(l_err);
    printf("mesh extracting\n");
    oc.mesh_extract(g_frame,l_err);
    //    oc.quality_improve(g_frame,1);
    printf("num_verts = %d\n",g_frame.numverts);
    printf("copy geoframe to Geometry\n");

    copyGeoframeToGeometry(g_frame,output);

    for(int i = 0; i < output->m_NumTriVerts; i++)
      {
	for(int j = 0; j < 3; j++)
	  output->m_TriVerts[i*3+j] = oc.minext[j] + oc.spans[j]*output->m_TriVerts[i*3+j];
      }

    return output;
  }

  boost::shared_ptr<Geometry> refineMesh(const boost::shared_ptr<Geometry>& input)
  {
    Octree oc;

    geoframe g_frame;
    boost::shared_ptr<Geometry> output/*(new Geometry())*/;
    printf("copy geometry to geframe\n");
    copyGeometryToGeoframe(input,g_frame);
    //oc.set_isovalue(DEFAULT::IVAL);
    //oc.set_isovalue_in(DEFAULT::IVAL_IN);

    printf("set volume\n");
    g_frame.setSpan(1.0,1.0,1.0);
    g_frame.setMin(0.0,0.0,0.0);
    oc.setMeshType(0);
    oc.setNormalType(0);
    printf("mesh update\n");
    oc.geometric_flow_tri(g_frame);
    printf("finish mesh update\n");

    copyGeoframeToGeometry(g_frame,output);
    return output;
  }
}

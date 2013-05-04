#ifndef MESH_IO_H
#define MESH_IO_H

#include <boost/shared_ptr.hpp>
#include <cvcraw_geometry/Geometry.h>

#include <multi_sdf/mds.h>

namespace multi_sdf
{

enum FILE_TYPE
{
   OFF,
   COFF,
   RAW,
   RAWN,
   RAWC,
   RAWNC,
   STL,
   SMF
};

void
read_labeled_mesh(Mesh &mesh, const string& ip_filename, 
                  const FILE_TYPE& ftype, 
                  const bool& read_color_opacity, 
                  const bool& is_uniform);

void
read_labeled_mesh(Mesh &mesh,
		  const boost::shared_ptr<Geometry>& geom);

void
write_mesh(const Mesh& mesh, const char* ofname, FILE_TYPE ftype, 
           bool write_color_opacity, bool use_input_mesh_color, 
           float r, float g, float b, float a);

}

#endif

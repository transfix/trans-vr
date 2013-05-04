//
// C++ Implementation: main
//
// Description: 
//
//
// Author:  <>, (C) 2010
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include <SweetMesh/hexmesh.h>
#include <SweetMesh/meshTools.h>
#include <SweetMesh/meshIO.h>
#include <SweetMesh/volRoverDisplay.h>

//main()============================
int main(int argc, char **argv){
	if(argc != 3){
		std::cerr << "Usage " << argv[0] << " <input_mesh.rawhs> <output_vis.linec>\n";
		return 1;
	}
}
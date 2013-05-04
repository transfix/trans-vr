/*
  Copyright 2005-2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolUtils.

  VolUtils is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolUtils is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: volinfo.cpp 5355 2012-04-06 22:16:56Z transfix $ */

#include <VolMagick/VolMagick.h>

#include <boost/format.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/join.hpp>

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <fstream>

// 04/06/2012 - transfix - added bounding box modification code
int main(int argc, char **argv)
{
  using namespace std;

  try
    {
      switch(argc)
        {
        case 8:
          {
            using namespace boost;
            vector<string> parts;
            for(int i = 0; i < 6; i++)
              parts.push_back(argv[i+2]);
            VolMagick::BoundingBox bbox(join(parts,","));
            VolMagick::writeBoundingBox(bbox,argv[1]);
          }
        case 2:
          {
            VolMagick::VolumeFileInfo volinfo;
            volinfo.read(argv[1]);
            cout << volinfo.filename() << ":" <<endl;
            cout << "Num Variables: " << volinfo.numVariables() << endl;
            cout << "Num Timesteps: " << volinfo.numTimesteps() << endl;
            cout << "Dimension: " << volinfo.XDim() << "x" << volinfo.YDim() << "x" << volinfo.ZDim() << endl;
            cout << "Bounding box: ";
            cout << "(" << volinfo.boundingBox().minx << "," << volinfo.boundingBox().miny << "," << volinfo.boundingBox().minz << ") ";
            cout << "(" << volinfo.boundingBox().maxx << "," << volinfo.boundingBox().maxy << "," << volinfo.boundingBox().maxz << ") ";
            cout << endl;
            cout << "Span: " << "(" << volinfo.XSpan() << "," << volinfo.YSpan() << "," << volinfo.ZSpan() << ") " << endl;
            double volmin = volinfo.min(), volmax = volinfo.max();
            cout<<"volhead info: " << volmin <<" " << volmax<< endl;
            for(unsigned int i = 0; i<volinfo.numVariables(); i++)
              {
                cout << "Name of var " << i << ": " << volinfo.name(i) << endl;
                cout << "Voxel type of var " << i << ": " << volinfo.voxelTypeStr(i) << endl;
                for(unsigned int j = 0; j<volinfo.numTimesteps(); j++)
                  {
                    if(volmin > volinfo.min(i,j)) volmin = volinfo.min(i,j);
                    if(volmax < volinfo.max(i,j)) volmax = volinfo.max(i,j);
                    cout << "Min voxel value of var " << i << ", timestep " << j << ": " << volinfo.min(i,j) << endl;
                    cout << "Max voxel value of var " << i << ", timestep " << j << ": " << volinfo.max(i,j) << endl;
                  }
              }
            cout << "Min voxel value (of whole dataset): " << volmin << endl;
            cout << "Max voxel value (of whole dataset): " << volmax << endl;
            break;
          }
        default:
          {
            cerr << "Usage: " << endl;
            cerr << argv[0] << " <volume file>" << endl;
            cerr << argv[0] << " <volume file> [minx] [miny] [minz] [maxx] [maxy] [maxz] : set a volume's bounding box" << endl;
            return 1;
          }
        }
    }
  catch(VolMagick::Exception &e)
    {
      cerr << e.what() << endl;
    }
  catch(std::exception &e)
    {
      cerr << e.what() << endl;
    }

  return 0;
}

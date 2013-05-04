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

#include <iostream>
#include <VolMagick/VolMagick.h>
#include <stdlib.h>

int main(int argc,char *argv[])
{
	if(argc<2)
	{
		std::cout<<"Usage: volextract <input rawvfile> <output rawiv file> <variable index> <timestep index>\n";
		return 0;
	}
	int varindex=atoi(argv[3]);
	int timeindex=atoi(argv[4]);
	VolMagick::Volume inVolume;
	std::cout<<"Reading file: "<<argv[1]<<std::endl;
	VolMagick::readVolumeFile(inVolume,argv[1],varindex,timeindex);
	std::cout<<"Extracting info for varindex: "<<varindex<<" timeindex: "<<timeindex<<std::endl;
	std::cout<<"Writing file: "<<argv[2]<<std::endl;
	VolMagick::createVolumeFile(inVolume,argv[2]);
	return 0;
}

	

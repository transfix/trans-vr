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
#include <vector>
#include <map>
#include "VolMagick/VolMagick.h"

using namespace std;

int main(int argc, char *argv[])
{
	if(argc<3)
	{
		cout<<argv[0]<<" <input volume> <output volume>"<<endl;
		return 1;
	}

	//Read the rawiv; count the number of unique values.
	VolMagick::Volume vol;
	VolMagick::readVolumeFile(vol,argv[1]);
	
	int x=vol.XDim(),y=vol.YDim(),z=vol.ZDim();
	
	VolMagick::Volume newvol(VolMagick::Dimension(x,y,z),VolMagick::Float,vol.boundingBox());

	vector<double> unique;
	double value;

	map<int, int> myMap;
	map<int,int>::iterator it;

  //  cout<<"vol[1]:" << vol(1,1,1) <<endl;
	// Go through the data set, checking every value. If the value is not in unique, add to unique.
	for(int i=0;i<x;i++)
		for(int j=0;j<y;j++)
			for(int k=0;k<z;k++)
			{
				value=vol(i,j,k);
				int l=0;
				for(;l<unique.size();l++)
					if(unique[l]==value) break;
				if(l==unique.size())
				{
					unique.push_back(value);
				//	myMap[value] = l;
				}
			}
	cout<<"Unique values found are: ";
	for(int i=0;i<unique.size();i++) 
	{
	cout<<unique[i]<<" ";
	myMap[unique[i]]=i;
	}
	cout<<endl;

//	cout<<"mymap size "<<myMap.size() <<endl;
//	for(it=myMap.begin(); it!=myMap.end(); ++it)
//	cout<<(*it).first<<" "<<(*it).second <<" "<<endl;
//	cout << endl;

	//Now go through the data again, and this time replace the values.
	for(int i=0;i<x;i++)
		for(int j=0;j<y;j++)
			for(int k=0;k<z;k++)
			{

			/*	value=vol(i,j,k);
				int l=0;
				for(;l<unique.size();l++)
					if(unique[l]==value) break;
				if(value!=0)
					newvol(i,j,k, (l+1));
				else
					newvol(i,j,k,0); */

				value = vol(i,j,k);
			//	if(value >= unique.size()) newvol(i,j,k, 0);
			//	else newvol(i,j,k, value+1);
				newvol(i,j,k, myMap[value]);
			}


	//Write the file out
	VolMagick::createVolumeFile(newvol,argv[2]);  
	cout<<"done !"<<endl;
}


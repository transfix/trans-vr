/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of SecondaryStructures.

  SecondaryStructures is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  SecondaryStructures is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <Histogram/histogram_data.h>
#include <map>
#include <SecondaryStructures/helix.h>
#include <SecondaryStructures/op.h>

using namespace SecondaryStructures;

#define THRESHOLD 4 //3 //2
#define SUBDIV 100
#define RADIUS 1.8
#define HELIX_LENGTH_THRESHOLD 3


extern "C" {void dsyev_( char* jobz, char* uplo, int* n, double* a, int* lda,
                double* w, double* work, int* lwork, int* info );
				}
#define N 3
#define LDA N



//Rewrite this function to build better cylinders  fitting each component with a straight line.
vector< vector<Point> > build_cylindrical_helices(const vector< vector<int> >& curve_comps, const Curve& curve, ofstream& fout)
{
	vector< vector <Point> > helices;
	vector<Cylinder> cyls;
	vector<double> lengths;
	fout<<"{LIST" << endl;
	for(int i=0; i< (int) curve_comps.size(); i++)
	{
		Vector centold(0.0, 0.0, 0.0); 

		for(int j=0; j < (int) curve_comps[i].size(); j++)
		{
			Point p1 = curve.vert_list[curve_comps[i][j]].pos;
			Vector tt = p1-CGAL::ORIGIN;

			centold = centold + tt;
		}
		centold = centold/((double)(curve_comps[i].size()));

		Point cent(centold.x(), centold.y(), centold.z());

		double sumXX = 0.0;
		double sumXY = 0.0;
		double sumXZ = 0.0;
		double sumYY = 0.0;
		double sumYZ = 0.0;
		double sumZZ = 0.0;

		for (int j = 0; j < (int) curve_comps[i].size(); j++)
		{
			Point p1 = curve.vert_list[curve_comps[i][j]].pos;
//			Vector tt = p1 - CGAL::ORIGIN;
			Vector diff = p1 - cent;
			sumXX += diff.x()*diff.x();
       		sumXY += diff.x()*diff.y();
	       	sumXZ += diff.x()*diff.z();
	        sumYY += diff.y()*diff.y();
    	   	sumYZ += diff.y()*diff.z();
	        sumZZ += diff.z()*diff.z();
		}
		sumXX /= (double)(curve_comps[i].size());
		sumXY /= (double)(curve_comps[i].size());
		sumXZ /= (double)(curve_comps[i].size());
		sumYY /= (double)(curve_comps[i].size());
		sumYZ /= (double)(curve_comps[i].size());
		sumZZ /= (double)(curve_comps[i].size());
	
		double eigensys[3][3];
		eigensys[0][0] = (double) (sumYY + sumZZ);
		eigensys[0][1] = (double)(-sumXY);
		eigensys[0][2] = (double)(-sumXZ);
		eigensys[1][0] = eigensys[0][1];
		eigensys[1][1] = (double)(sumXX + sumZZ);
		eigensys[1][2] = (double)(-sumYZ);
		eigensys[2][0] = eigensys[0][2];
		eigensys[2][1] = eigensys[1][2];
		eigensys[2][2] = (double)(sumXX + sumYY);

		int n = N, lda = LDA, info, lwork;
		double wkopt;
		double* work;
		double w[N];
		double a[LDA*N] ={ eigensys[0][0], 0.00, 0.00,
						   eigensys[0][1], eigensys[1][1], 0.0,
						   eigensys[0][2], eigensys[1][2], eigensys[2][2]};
	    lwork = -1;
		dsyev_("Vectors", "Upper", &n, a, &lda, w, &wkopt, &lwork, &info);
		lwork = (int)wkopt;
		work = (double*)malloc(lwork*sizeof(double));
		dsyev_("Vectors", "Upper", &n, a, &lda, w, work, &lwork, &info);
		if( info > 0 ) {
        	        printf( "The algorithm failed to compute eigenvalues.\n" );
            	    exit( 1 );
	    }
		else 
		{   //w[0] is the minimal eignevalue, a[0-2] is corresponding unit eigen vector
			
			// the line is cent + \lambda * a[0-2];
//			cout<<"eigenvalue is  "<< w[0] <<"  " << w[1] <<"  "<<w[2] <<endl;
//			for(int t=0; t<3; t++)
//			cout<<a[t]<<" " <<a[t+3]<<" "<<a[t+6]<<endl;
			double lambda[curve_comps[i].size()];
			Vector tempp;
			int j = 0;
			for(int k=0; k < curve_comps[i].size(); k++)
			{
				Point p1 = curve.vert_list[curve_comps[i][k]].pos;
	//			Vector tt = p1 - CGAL::ORIGIN;
				tempp = p1 - cent;
				lambda[j] = tempp.x()*a[0] + tempp.y()*a[1] + tempp.z() * a[2];
				j++;
			}

			double max = lambda[0];
			double min = lambda[0];
			for(int k = 1; k < curve_comps[i].size(); k++)
			{
				max = (max < lambda[k])?lambda[k]:max;
				min = (min > lambda[k])?lambda[k]:min;
			}
			
	//		cout<<" max, min: " << max <<" " << min << endl;
			double tem1[3], tem2[3];
			tem1[0] = max*a[0] + centold.x();
			tem1[1] = max*a[1] + centold.y();
			tem1[2] = max*a[2] + centold.z();

			tem2[0] = min*a[0] + centold.x();
			tem2[1] = min*a[1] + centold.y();
			tem2[2] = min*a[2] + centold.z();

			Point left(tem1[0], tem1[1], tem1[2]), right(tem2[0], tem2[1],tem2[2]);

			vector <Point> V;
			V.push_back(left);
			V.push_back(right);
			Vector aa = right - left;

			if (sqrt(CGAL::to_double((right-left)*(right-left))) < HELIX_LENGTH_THRESHOLD)
			{
				continue;
			}

			cyls.push_back(Cylinder(V,aa));
			lengths.push_back(CGAL::to_double((left-right)*(left-right)));

			//srand48(i);
			//double red = drand48(), green = drand48(), blue = drand48();
	//float r = drand48(), g = drand48(), b = drand48();
			srand(i);
			double 
			  red = double(rand())/double(RAND_MAX), 
			  green = double(rand())/double(RAND_MAX), 
			  blue = double(rand())/double(RAND_MAX);
			
			fout<<"{OFF" << endl;
			fout<<" 2 1 0 " << endl;
			fout<< left.x() <<" " << left.y() <<" " << left.z() << endl;
			fout<< right.x() <<" " << right.y() <<" " << right.z() << endl;
			fout<<"2   0 1  " <<red <<" " << green <<" " << blue << " 1"<< endl;
			fout<<"}" << endl;
		}
	}
	fout<<"}"<<endl;

		// insertion sort (will sort by length key)
	std::map<double,Cylinder> cylsSorted;
	for (int i=0; i<cyls.size(); i++)
	{
		double key = lengths[i];
		Cylinder value = cyls[i];
		cylsSorted.insert(std::make_pair(key,value));
	}
	// write the cylinders as mesh.
	// for(map<double,Cylinder>::iterator i = cylsSorted.begin(); i != cylsSorted.end(); i++)
	int counter = 0;
	map<double,Cylinder>::iterator i = cylsSorted.end();
	while (i != cylsSorted.begin())
	{
		i--;
		counter++;
		// 2-step process: first create SUBDIV number of points on a circle.
		// Point v0 = cyls[i].first[0];
		// Vector a = cyls[i].second;
		Point v0 = i->second.first[0];
		Vector a = i->second.second;
		vector<Vector> circle;
		for (int j = 0; j < SUBDIV; j ++)
		{
			double theta = 2*M_PI/SUBDIV*j;
			double x = RADIUS*cos(theta),
				   y = RADIUS*sin(theta),
				   z = 0;
			circle.push_back(Vector(x,y,z));
		}
		// rotate and translate the circle to be aligned with a
		// and centered at v0.
		// translation vector 't'.
		// rotation parameters: axis 'raxis' passing through origin.
		// translation parameters.
		Vector t = v0 - CGAL::ORIGIN;
		// rotation parameters.
		Vector zaxis(0,0,1);
		Vector raxis = CGAL::cross_product(zaxis,a);
		raxis = (1./sqrt(CGAL::to_double(raxis*raxis))) * raxis;
		double cos_sq = CGAL::to_double(zaxis*a)/sqrt(CGAL::to_double(a*a));
		double sin_sq = sqrt(1-cos_sq*cos_sq);
		vector<Point> pts1; // points on lower rim.
		transform(pts1, circle, t, CGAL::ORIGIN, raxis, cos_sq, sin_sq);
		vector<Point> pts2; // points on upper rim.
		transform(pts2, circle, t+a, CGAL::ORIGIN, raxis, cos_sq, sin_sq);
		pts2.insert(pts2.begin(), pts1.begin(), pts1.end()); // pad the points of pts1 in front of pts2.
		pts2.push_back(v0);
		pts2.push_back(v0+a);
		helices.push_back(pts2); // push larger(later) values in front
	}
	assert(counter == cylsSorted.size());
	
	return helices;

}

/*
vector< vector<Point> > build_cylindrical_helices(const vector< vector<int> >& curve_comps, const Curve& curve, ofstream& fout)
{
	vector< vector<Point> > helices; // each sub-vector contains 2k points,
	// k on each rim of the cylinder.
	// for every curve component, find the furthest pair of points.
	vector<Cylinder> cyls; // holding the axis and endpoints of every cylinder temporarily.
	vector<double> lengths;
	fout << "{LIST" << endl;
	for (int i = 0; i < (int)curve_comps.size(); i ++)
	{
		int ep[2] = {-1,-1};
		double max = -HUGE;
		if ((int)curve_comps[i].size() < 2)
		{
			continue;
		}
		for (int j = 0; j < (int)curve_comps[i].size(); j ++)
		{
			Point p1 = curve.vert_list[curve_comps[i][j]].pos;
			for (int k = j+1; k < (int)curve_comps[i].size(); k ++)
			{
				Point p2 = curve.vert_list[curve_comps[i][k]].pos;
				double d = CGAL::to_double((p1-p2)*(p1-p2));
				if (d > max)
				{
					max = d;
					ep[0] = curve_comps[i][j];
					ep[1] = curve_comps[i][k];
				}
			}
		}
		CGAL_assertion(ep[0] != -1 && ep[1] != -1);
		Point v0 = curve.vert_list[ep[0]].pos;
		Point v1 = curve.vert_list[ep[1]].pos;
		// check the length of this helix.
		if (sqrt(CGAL::to_double((v0-v1)*(v0-v1))) < HELIX_LENGTH_THRESHOLD)
		{
			continue;
		}
		vector<Point> V;
		V.push_back(v0);
		V.push_back(v1);
		Vector a = v1-v0;
		cyls.push_back(Cylinder(V,a));
		lengths.push_back(CGAL::to_double((v0-v1)*(v0-v1)));
		// debug
		// write all the points in the curve component.
		//srand48(i);
		//double red = drand48(), green = drand48(), blue = drand48();
		
		srand(i);
		double 
		red = double(rand())/double(RAND_MAX), 
		green = double(rand())/double(RAND_MAX), 
		blue = double(rand())/double(RAND_MAX);
		fout << "{ appearance {linewidth 4}" << endl;
		fout << "OFF" << endl;
		fout << (int)curve_comps[i].size() << " "
			 << (int)curve_comps[i].size() << " 0" << endl;
		for (int j = 0; j < (int)curve_comps[i].size(); j ++)
		{
			fout << curve.vert_list[curve_comps[i][j]].pos << endl;
		}
		for (int j = 0; j < (int)curve_comps[i].size(); j ++)
		{
			fout << "1\t" << j << " " << red << " " << green << " " << blue << " 1" << endl;
		}
		fout << "}" << endl;
		// end debug
	}
	fout << "}" << endl;
	// insertion sort (will sort by length key)
	std::map<double,Cylinder> cylsSorted;
	for (int i=0; i<cyls.size(); i++)
	{
		double key = lengths[i];
		Cylinder value = cyls[i];
		cylsSorted.insert(std::make_pair(key,value));
	}
	// write the cylinders as mesh.
	// for(map<double,Cylinder>::iterator i = cylsSorted.begin(); i != cylsSorted.end(); i++)
	int counter = 0;
	map<double,Cylinder>::iterator i = cylsSorted.end();
	while (i != cylsSorted.begin())
	{
		i--;
		counter++;
		// 2-step process: first create SUBDIV number of points on a circle.
		// Point v0 = cyls[i].first[0];
		// Vector a = cyls[i].second;
		Point v0 = i->second.first[0];
		Vector a = i->second.second;
		vector<Vector> circle;
		for (int j = 0; j < SUBDIV; j ++)
		{
			double theta = 2*M_PI/SUBDIV*j;
			double x = RADIUS*cos(theta),
				   y = RADIUS*sin(theta),
				   z = 0;
			circle.push_back(Vector(x,y,z));
		}
		// rotate and translate the circle to be aligned with a
		// and centered at v0.
		// translation vector 't'.
		// rotation parameters: axis 'raxis' passing through origin.
		// translation parameters.
		Vector t = v0 - CGAL::ORIGIN;
		// rotation parameters.
		Vector zaxis(0,0,1);
		Vector raxis = CGAL::cross_product(zaxis,a);
		raxis = (1./sqrt(CGAL::to_double(raxis*raxis))) * raxis;
		double cos_sq = CGAL::to_double(zaxis*a)/sqrt(CGAL::to_double(a*a));
		double sin_sq = sqrt(1-cos_sq*cos_sq);
		vector<Point> pts1; // points on lower rim.
		transform(pts1, circle, t, CGAL::ORIGIN, raxis, cos_sq, sin_sq);
		vector<Point> pts2; // points on upper rim.
		transform(pts2, circle, t+a, CGAL::ORIGIN, raxis, cos_sq, sin_sq);
		pts2.insert(pts2.begin(), pts1.begin(), pts1.end()); // pad the points of pts1 in front of pts2.
		helices.push_back(pts2); // push larger(later) values in front
	}
	assert(counter == cylsSorted.size());
	return helices;
}
*/

double dist(Point p1, Point p2, Point p3, Point p4)
{
	double d1 = CGAL::to_double((p1-p3).squared_length());
	double d2 = CGAL::to_double((p1-p4).squared_length());
	double d3 = CGAL::to_double((p2-p3).squared_length());
	double d4 = CGAL::to_double((p2-p4).squared_length());
 
   if(d1 <= d2 && d1 <= d3 && d1 <= d4) return sqrt(d1);
   else if ( d2 <= d1 && d2 <= d3 && d2 <= d4) return sqrt(d2);
   else if ( d3 <= d1 && d3 <= d3 && d3 <= d4) return sqrt(d3);
   else return sqrt(d4);
}



vector< vector<int> > cluster(const vector<Point>& hpts, Curve& curve)
{
	for (int i = 0; i < (int)hpts.size(); i ++)
	{
		curve.vert_list.push_back(CVertex(hpts[i]));
		curve.vert_list[(int)curve.vert_list.size()-1].id = (int)curve.vert_list.size()-1;
		curve.vert_list[(int)curve.vert_list.size()-1].visited = false;
	}
	for (int i = 0; i < (int)curve.vert_list.size(); i ++)
	{
		Point pi = curve.vert_list[i].pos;
		for (int j = i+1; j < (int)curve.vert_list.size(); j ++)
		{
			Point pj = curve.vert_list[j].pos;
			// if i-th and j-th points are within a threshold, connect them.
			if (sqrt(CGAL::to_double((pi-pj)*(pi-pj))) < THRESHOLD)
			{
				curve.edge_list.push_back(CEdge(i,j));
				curve.vert_list[i].inc_vid_list.push_back(j);
				curve.vert_list[j].inc_vid_list.push_back(i);
			}
		}
	}
	// walk to collect the connected components.
	vector< vector<int> > curve_comps;
	vector< vector<CEdge> > edge_comps;
	vector< vector<int> > curve_comps_temp;
	for (int i = 0; i < (int)curve.vert_list.size(); i ++)
	{
		CVertex v = curve.vert_list[i];
		if (v.visited)
		{
			continue;
		}
		vector<int> Q;
		Q.push_back(i);
		curve.vert_list[i].visited = true;
		vector<int> new_comp;
		vector <CEdge> new_edge;
		while (! Q.empty())
		{
			int vid = Q.back();
			Q.pop_back();
			new_comp.push_back(vid);
			// add the unvisited incident vertices to Q.
			for (int j = 0; j < (int)curve.vert_list[vid].inc_vid_list.size(); j ++)
			{
				int uid = curve.vert_list[vid].inc_vid_list[j];
				if (curve.vert_list[uid].visited)
				{
					continue;
				}
				Q.push_back(uid);
				new_edge.push_back(CEdge(vid,uid));
				curve.vert_list[uid].visited = true;
			}
		}
		curve_comps_temp.push_back(new_comp);
		edge_comps.push_back(new_edge);
	}

#ifdef DEBUG_OP
   
    ofstream fout;
	fout.open("temp.off");
	fout<<"{LIST " << endl;

    for(int i = 0; i< edge_comps.size(); i++)
	{
		vector<int> uid;
		vector<int> vid;

		for(int j = 0; j < edge_comps[i].size(); j++)
		{
			uid.push_back(edge_comps[i][j].ep[0]);
			vid.push_back(edge_comps[i][j].ep[1]);
		}
		fout<<"{OFF" << endl;
		fout<<(edge_comps[i].size()*2) << " " << edge_comps[i].size() <<" 0" << endl;
		for(int j = 0; j < uid.size(); j++)
		{
			fout<<curve.vert_list[uid[j]].pos << endl;
			fout<<curve.vert_list[vid[j]].pos << endl;
		}
		
		//srand48(i);
		//double red = drand48(), green = drand48(), blue = drand48();
		srand(i);
		double 
		  red = double(rand())/double(RAND_MAX), 
		  green = double(rand())/double(RAND_MAX), 
		  blue = double(rand())/double(RAND_MAX);
		
		for(int j = 0; j < uid.size(); j++)
		{
			fout <<"2 " << (2*j) << " " << (2*j+1) <<" "<< red<<" " << green<<" " << blue<<" 1"<<endl;
		}
		fout<<"}"<<endl;
	}
	fout<<"}"<<endl;
	fout.close();

#endif
     //using the neighbour hood infor to do refine 


    vector< vector<CEdge> > edge_comps_refine;
	//refine edge_comps_temp 
	for(int i = 0; i < edge_comps.size(); i++)
	{

		if(edge_comps[i].size() <=1)
		{
			vector <CEdge> temp_comp;
			for(int j = 0; j < edge_comps[i].size(); j++)
			{
				temp_comp.push_back(edge_comps[i][j]);
			}
			edge_comps_refine.push_back(temp_comp);
			continue;
		}

		vector <CEdge> temp_compa;
		vector <CEdge> temp_compb;
		vector <CEdge> temp_compc;
		for(int j = 0; j < edge_comps[i].size(); j++)
		{	
			temp_compa.push_back(edge_comps[i][j]);
		}
		while(temp_compa.size()>1)
		{
			int vid = temp_compa.front().ep[0];
			int uid = temp_compa.front().ep[1];

		 	temp_compb.push_back(temp_compa.front());
			temp_compa.erase(temp_compa.begin());

			Point p1 = curve.vert_list[vid].pos;
			Point p2 = curve.vert_list[uid].pos;
			Vector v1 = p2-p1;


			for(int j = 0; j<temp_compa.size(); j++)
			{
				int vida = temp_compa[j].ep[0];
				int uida = temp_compa[j].ep[1];

				Point p3 = curve.vert_list[vida].pos;
				Point p4 = curve.vert_list[uida].pos;
				Vector v2 = p4 - p3;
				bool flag = false;
				if( fabs((v1*v2)*(v1*v2)/CGAL::to_double(v1.squared_length()*v2.squared_length()))>=0.75  && dist(p1,p2,p3,p4) < THRESHOLD) 
				{
					temp_compb.push_back(temp_compa[j]);
					v1 = v2;
					p1 = p3;
					p2 = p4;
					flag = true;
				}
				else 
				{
					for(int l = 0; l < temp_compb.size(); l++)
					{
						int vtem = temp_compb[l].ep[0];
						int utem = temp_compb[l].ep[1];

						Point p5 = curve.vert_list[vtem].pos;
						Point p6 = curve.vert_list[utem].pos;

						Vector v3 = p6-p5;
						if(fabs((v3*v2)*(v3*v2)/CGAL::to_double(v3.squared_length()*v2.squared_length()))>= 0.75 && dist(p3,p4,p5,p6) < THRESHOLD)
						{ 
//							v1 = v3;
//							p1 = p5;
//							p2 = p6;
						    flag  = true;
							temp_compb.push_back(temp_compa[j]);
							break;
						}
					}
				}
				if(!flag)
				temp_compc.push_back(temp_compa[j]);
			}
/*
			int vid = temp_compa.back().ep[0];
			int uid = temp_compa.back().ep[1];

		 	temp_compb.push_back(temp_compa.back());
			temp_compa.pop_back();

			Point p1 = curve.vert_list[vid].pos;
			Point p2 = curve.vert_list[uid].pos;
			Vector v1 = p2-p1;


			for(int j = temp_compa.size()-1; j>=0; j--)
			{
				int vida = temp_compa[j].ep[0];
				int uida = temp_compa[j].ep[1];

				Point p3 = curve.vert_list[vida].pos;
				Point p4 = curve.vert_list[uida].pos;
				Vector v2 = p4 - p3;
				bool flag = false;
				if( fabs((v1*v2)*(v1*v2)/CGAL::to_double(v1.squared_length()*v2.squared_length()))>=0.75  && dist(p1,p2,p3,p4) < THRESHOLD) 
				{
					temp_compb.push_back(temp_compa[j]);
					v1 = v2;
					p1 = p3;
					p2 = p4;
					flag = true;
				}
				else 
				{
					for(int l = 0; l < temp_compb.size(); l++)
					{
						int vtem = temp_compb[l].ep[0];
						int utem = temp_compb[l].ep[1];

						Point p5 = curve.vert_list[vtem].pos;
						Point p6 = curve.vert_list[utem].pos;

						Vector v3 = p6-p5;
						if(fabs((v3*v2)*(v3*v2)/CGAL::to_double(v3.squared_length()*v2.squared_length()))>= 0.75 && dist(p3,p4,p5,p6) < THRESHOLD)
						{ 
							v1 = v3;
							p1 = p5;
							p2 = p6;
						    flag  = true;
							temp_compb.push_back(temp_compa[j]);
							break;
						}
					}
				}
				if(!flag)
				temp_compc.push_back(temp_compa[j]);
			} */


			temp_compa.clear();
			if(temp_compc.size()>2) temp_compa = temp_compc;
			temp_compc.clear();
			edge_comps_refine.push_back(temp_compb);
			temp_compb.clear();

		}

	} 
#ifdef DEBUG_OP
   // ofstream fout;
	fout.open("tempN.off");
	fout<<"{LIST " << endl;

    for(int i = 0; i< edge_comps_refine.size(); i++)
	{
		vector<int> uid;
		vector<int> vid;

		for(int j = 0; j < edge_comps_refine[i].size(); j++)
		{
			uid.push_back(edge_comps_refine[i][j].ep[0]);
			vid.push_back(edge_comps_refine[i][j].ep[1]);
		}
		fout<<"{OFF" << endl;
		fout<<(edge_comps_refine[i].size()*2) << " " << edge_comps_refine[i].size() <<" 0" << endl;
		for(int j = 0; j < uid.size(); j++)
		{
			fout<<curve.vert_list[uid[j]].pos << endl;
			fout<<curve.vert_list[vid[j]].pos << endl;
		}
		
		//srand48(i);
		//double red = drand48(), green = drand48(), blue = drand48();
		srand(i);
		double 
		  red = double(rand())/double(RAND_MAX), 
		  green = double(rand())/double(RAND_MAX), 
		  blue = double(rand())/double(RAND_MAX);

		for(int j = 0; j < uid.size(); j++)
		{
			fout <<"2 " << (2*j) << " " << (2*j+1) <<" "<< red<<" " << green<<" " << blue<<" 1"<<endl;
		}
		fout<<"}"<<endl;
	}
	fout<<"}"<<endl;
	fout.close();
#endif

	// walk to collect the connected components.
 	for(int i = 0; i < edge_comps_refine.size(); i++)
	{
		vector <int> temp_comp;
		set <int> tempset;
		set <int>::iterator k;
		for(int j = 0; j < edge_comps_refine[i].size(); j++)
		{
			int uid = edge_comps_refine[i][j].ep[0];
			int vid = edge_comps_refine[i][j].ep[1];
			tempset.insert(uid);
			tempset.insert(vid);
		}
		for(k = tempset.begin(); k!= tempset.end(); ++k)
		   temp_comp.push_back(*k);
		curve_comps.push_back(temp_comp);
		temp_comp.clear();
		tempset.clear();
		
	}
	

	return curve_comps;
}




// get a list of valid edges to construct histogram data from
vector<SEdge> Skel::get_valid_edge_list()
{
	// this vector holds the edges as selected by the user with the histogram
	vector<SEdge> valid_edge_list;
	for (int i = 0; i < get_ne(); i ++)
	{
		// ignore conditions:
		// 1. more than one included face
		// 2. endpoints in same component
		if (edge_list[i].num_inc_face >= 1)
		{
			continue;
		}
		else
		{
			if (in_same_comp(edge_list[i].get_endpoint(0),
							 edge_list[i].get_endpoint(1)))
			{
				continue;
			}
		}
		// this edge passes.
		valid_edge_list.push_back(edge_list[i]);
	}
	return valid_edge_list;
}

HistogramData Skel::get_alpha_histogram_data()
{
	// create histogram data
	const int n_bins = 128;
	vector<double> widths;
	vector<SEdge> valid_edge_list = get_valid_edge_list();
	for (int i=0; i<valid_edge_list.size(); i++)
	{
		widths.push_back(valid_edge_list[i].width);
	}
	return HistogramData(widths, n_bins);
}


HistogramData Skel::get_beta_histogram_data()
{
	// create histogram data
	const int n_bins = 128;
	vector<double> widths;
	vector<SFace> valid_face_list = face_list;
	for (int i=0; i<valid_face_list.size(); i++)
	{
		widths.push_back(valid_face_list[i].width);
	}
	return HistogramData(widths, n_bins);
}

// Pass the helix candidate points as a list of points.
void Skel::compute_helices(const char* file_prefix)
{
	helices.clear();
	// collect the points from the linear portion of the skeleton.
	vector<SEdge> valid_edge_list = get_valid_edge_list();
	vector<SEdge> user_selected_edge_list;
	// create user_selected_edge_list
	// this vector holds the edges as selected by the user with the histogram
	for (int i=0; i< valid_edge_list.size(); i++)
	{
		// prune by width  //Not a good way to do this.
		if (valid_edge_list[i].width < _alphaMinWidth ||
				valid_edge_list[i].width > _alphaMaxWidth)
		{
			continue;
		}
		// this edge passes
		user_selected_edge_list.push_back(valid_edge_list[i]);
	}
	// pass user data to algorithm
	set<int> hvids; // id of the vertices on L which are selected by width cut-off.
	for (int i = 0; i < user_selected_edge_list.size(); i++)
	{
		hvids.insert(user_selected_edge_list[i].get_endpoint(0));
		hvids.insert(user_selected_edge_list[i].get_endpoint(1));
	}
	if (hvids.empty())
	{
		return;
	}
	vector<Point> hpts;
	for (set<int>::iterator it = hvids.begin();
			it != hvids.end(); it++)
	{
		hpts.push_back(vert_list[(*it)].point());
	}
	// collect the components of the helical region.
	Curve curve;
	vector< vector<int> > curve_comps = cluster(hpts, curve);

	std::cout<<"curve comp: " << curve_comps.size() << std::endl;
	// for every curve component, fit a cylinder.
	ofstream fout;
	char filename[1000];
	strcat(strcpy(filename, file_prefix), ".curve");
	fout.open(filename);
	helices = build_cylindrical_helices(curve_comps, curve, fout);
	
	// save so that we can quickly rebuild point geometry
	_curve_comps = curve_comps;
	_curve = curve;
}

cvcraw_geometry::cvcgeom_t* Skel::buildHelixGeometry()
{
   
	// write_curve(curve, curve_comps, cylinders, output_file_prefix);
	// write the helix_cylinders in an off file.
	// create a TexMol geometry file for the helices
	std::vector<float> vertices;
	std::vector<unsigned int> indices;
	std::vector<float> colors;
	// for(int helix =0; helix<helices.size(); helix++) {
	 std::cerr << helix_cnt << "\t" << helices.size() << "\n";
	assert(helix_cnt <= helices.size());
	for (int helix =0; helix<helix_cnt; helix++)
	{
		vector<Point> points = helices[helix];
		// copy points to vertex vector
		assert(vertices.size()%3 == 0);
		assert((vertices.size()/3) == (helix * (SUBDIV + 1) * 2));
		const int startVertex = vertices.size()/3;
		assert(points.size() == (SUBDIV*2)+2);
		const float RED  = 0.99;
		const float GREEN  = 0.0;
		const float BLUE  = 0.7;
		const float ALPHA = 1;
		for (int i=0; i<points.size(); i++)
		{
			vertices.push_back(points[i].x());
			vertices.push_back(points[i].y());
			vertices.push_back(points[i].z());
			colors.push_back(RED);
			colors.push_back(GREEN);
			colors.push_back(BLUE);
	//		colors.push_back(ALPHA);
		}
		// build triangle array.  iterate along bottom rim

		//lower center
		const int lower_cent = startVertex + SUBDIV*2;
		//upper center
		const int upper_cent = startVertex + SUBDIV*2 + 1;

		for (int i=0; i<SUBDIV; i++)
		{
			// next_i follows i in the circular list
			int next_i = i+1;
			if (next_i == SUBDIV)
			{
				next_i = 0;
			}
			// this index, on lower rim
			const int lower_i = startVertex + i;
			// next index, on lower rim
			const int lower_next_i = startVertex + next_i;
			// this index, on upper rim
			const int upper_i = startVertex + i + SUBDIV;
			// next index, on upper rim
			const int upper_next_i = startVertex + next_i + SUBDIV;
            
			// tri 1
			indices.push_back(lower_i);
			indices.push_back(lower_next_i);
			indices.push_back(upper_i);
			// tri 2
			indices.push_back(lower_next_i);
			indices.push_back(upper_next_i);
			indices.push_back(upper_i);

			// tri 3 for bottom patch
			indices.push_back(lower_i);
			indices.push_back(lower_cent);
			indices.push_back(lower_next_i);

			// tri 4 for top patch
			indices.push_back(upper_i);
			indices.push_back(upper_next_i);
			indices.push_back(upper_cent);
		}
	}
	
	
	    cout<<" vertices, indices  " << vertices.size() << " " << indices.size() << " " << colors.size() << endl;
	vectors_to_tri_geometry(vertices,indices,colors,helixGeom);

/*	FILE* fp =fopen("helix.rawc", "w");F
	fprintf(fp, "%d %d\n", helixGeom->m_NumTriVerts, helixGeom->m_NumTris);
	for(int i=0; i< helixGeom->m_NumTriVerts; i++)
	{
		fprintf(fp, "%f %f %f %f %f %f\n", helixGeom->m_TriVerts[3*i +0], helixGeom->m_TriVerts[3*i +1], helixGeom->m_TriVerts[3*i +2], helixGeom->m_TriVertColors[3*i+0],
		 helixGeom->m_TriVertColors[3*i+1], helixGeom->m_TriVertColors[3*i+2]); 
	}

	for(int i=0; i<helixGeom->m_NumTris; i++)
	{
		fprintf(fp, "%d %d %d\n", helixGeom->m_Tris[3*i+0], helixGeom->m_Tris[3*i+1], 
helixGeom->m_Tris[3*i+2]); 

	}
	fclose(fp); */
	write_helix_wrl(helixGeom, "helix.wrl"); 
   return helixGeom;
}

cvcraw_geometry::cvcgeom_t* Skel::buildCurveGeometry(const vector< vector<int> >& curve_comps, const Curve& curve)
{

	// don't shuffle colors!
	//srand48(0);
	// srand48(time(0));
	// edges
  srand(0);
	std::vector<float> vertices;
	std::vector<float> colors;
	for (int i = 0; i < (int)curve_comps.size(); i ++)
	{
	  //double red = drand48(), green = drand48(), blue = drand48();
	
		double 
		  red = double(rand())/double(RAND_MAX), 
		  green = double(rand())/double(RAND_MAX), 
		  blue = double(rand())/double(RAND_MAX);
		for (int j = 0; j < (int)curve_comps[i].size(); j ++)
		{
			Point p0 = curve.vert_list[curve_comps[i][j]].pos;
			vertices.push_back(p0.x());
			vertices.push_back(p0.y());
			vertices.push_back(p0.z());
			colors.push_back(red);
			colors.push_back(green);
			colors.push_back(blue);
		}
	}

	vectors_to_point_geometry(vertices,colors,curveGeom);
    return curveGeom;
}

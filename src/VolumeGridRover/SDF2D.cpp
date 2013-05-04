/*
  Copyright 2005-2008 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeGridRover.

  VolumeGridRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeGridRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <VolumeGridRover/SDF2D.h>

namespace SDF2D
{
  /* 
     Return the angle between two vectors on a plane 
     The angle is from vector 1 to vector 2, positive anticlockwise
     The result is between -pi -> pi 
  */
  static inline double Angle2D(double x1, double y1, double x2, double y2)
  { 
    double dtheta,theta1,theta2;
    theta1 = atan2(y1,x1);
    theta2 = atan2(y2,x2);
    dtheta = theta2 - theta1;
    while (dtheta > M_PI) dtheta -= 2*M_PI;
    while (dtheta < -M_PI) dtheta += 2*M_PI; 
    return(dtheta);
  }

  //multi polygon version... this is probably too slow :o
  Image signedDistanceFunction(const std::vector<Polygon_2>& polys,
			       const Dimension& dim,
			       const BoundingBox& bbox,
			       SignMethod sign_method,
			       DistanceMethod dist_method)
  {
    ImageIndex WIDTH = dim.get<0>(), HEIGHT=dim.get<1>();
    //Polygon_2 polygon(poly);
    //std::vector<Point2DWrapper> treeVertices;
    std::vector<Polygon_2> polygons(polys);
    std::vector<std::vector<Point2DWrapper> > all_treeVertices;
    double overallMaxDistance = -DBL_MAX;
    double overallMinDistance = DBL_MAX;
    Image distanceArray(boost::extents[WIDTH][HEIGHT]);
    CGAL::Timer timer;
    timer.start();

    //std::cerr << "polygons size: " << polygons.size() << std::endl;

    //initialize the distance image
    for(ImageIndex i = 0; i != WIDTH; i++)
      for(ImageIndex j = 0; j != HEIGHT; j++)
	distanceArray[i][j] = -DBL_MAX;

    // get treeVertices from polygon...
    // build the treeVertices (Point2D wrappers that connect Edge2Ds to the Points)
    for(std::vector<Polygon_2>::iterator polygon_iter = polygons.begin();
	polygon_iter != polygons.end();
	polygon_iter++)
      {
	Polygon_2 &polygon = *polygon_iter;
	std::vector<Point2DWrapper> treeVertices;

	if(!polygon.is_simple())
	  {
	    std::cerr << "Warning: skipping non-simple polygon!" << std::endl;
	    continue;
	    //throw NonSimplePolygonException();
	  }

	// FIX POLYGON WINDING
	if(polygon.orientation() == CGAL::COUNTERCLOCKWISE)
	  polygon.reverse_orientation();

	for(Polygon_2::Vertex_iterator i = polygon.vertices_begin();
	    i != polygon.vertices_end();
	    i++)
	  {
	    std::set<Edge2DPtr> edgeSet;
	    Point2DPtr ptr(new Point2D(i->x(),i->y()));

	    for(Polygon_2::Edge_const_iterator j = polygon.edges_begin();
		j != polygon.edges_end();
		j++)
	      if(j->source() == *i || j->target() == *i)
		edgeSet.insert(Edge2DPtr(new Edge2D(Point2DPtr(new Point2D(j->source().x(),j->source().y())),
						    Point2DPtr(new Point2D(j->target().x(),j->target().y())))));
	
	    treeVertices.push_back(Point2DWrapper(ptr,edgeSet));
	  }

	all_treeVertices.push_back(treeVertices);
      }

    //if(!polygon.is_simple())
    //  throw NonSimplePolygonException();

    // FIX POLYGON WINDING
    //if(polygon.orientation() == CGAL::COUNTERCLOCKWISE)
    //  polygon.reverse_orientation();
    
    //std::cerr << "all_treeVertices size: " << all_treeVertices.size() << std::endl;

    std::cerr << "Img dim: " << WIDTH << ", " << HEIGHT << "\n";
    std::cerr << "Bbox: (" << bbox.xmin() << ", " << bbox.ymin() << ") (" << bbox.xmax() << ", " << bbox.ymax() << ")\n";

    // USE THE TREE VERTICES
    for(std::vector<std::vector<Point2DWrapper> >::iterator treeVertices_iter = all_treeVertices.begin();
	treeVertices_iter != all_treeVertices.end();
	treeVertices_iter++)
      {
	std::vector<Point2DWrapper> &treeVertices = *treeVertices_iter;
	Polygon_2 &polygon = *(polygons.begin() + std::distance(all_treeVertices.begin(),treeVertices_iter));

	Tree tree(treeVertices.begin(), treeVertices.end());

	if(std::distance(all_treeVertices.begin(),treeVertices_iter)%10 == 0)
	  std::cerr << "Building distance image..." 
		    << (100.0 * (float)std::distance(all_treeVertices.begin(),treeVertices_iter)/
			(all_treeVertices.size()-1)) << "\r";

	for(ImageIndex x = 0; x < WIDTH; x++)
	  {
	    for(ImageIndex y = 0; y < HEIGHT; y++)
	      {
		// distance test against edges
		double distance = DBL_MAX;
		bool insideCurve = false;

		//get the point in object coordinates
		double p_x = bbox.xmin() + (double(x)/double(WIDTH-1))*(bbox.xmax()-bbox.xmin());
		double p_y = bbox.ymin() + (double(y)/double(HEIGHT-1))*(bbox.ymax()-bbox.ymin());

		//std::cerr << "p_x,p_y = " << p_x << "," << p_y << std::endl;

		if(sign_method == ANGLE_SUM)
		  {
		    /*
		      simple test to see if point is inside or outside curve
		      http://local.wasp.uwa.edu.au/~pbourke/geometry/insidepoly/
		    */
		    double angle = 0.0;
		    for(int i = 0; i < polygon.size(); i++)
		      angle += Angle2D(polygon.vertex(i).x() - p_x,
				       polygon.vertex(i).y() - p_y,
				       polygon.vertex((i+1)%polygon.size()).x() - p_x,
				       polygon.vertex((i+1)%polygon.size()).y() - p_y);
		    if(fabs(angle) < M_PI) insideCurve = false;
		    else insideCurve = true;
		  }
		else if(sign_method == COUNT_EDGE_INTERSECTIONS)
		  {
		    int segmentIntersectionsX = 0;
		    for(Polygon_2::Edge_const_iterator i = polygon.edges_begin();
			i != polygon.edges_end();
			i++)
		      {
			Edge2D edge(Point2DPtr(new Point2D(i->source().x(),i->source().y())),
				    Point2DPtr(new Point2D(i->target().x(),i->target().y())));
			if(edge.intersectsXAxisAlignedLine(p_x,p_y,HEIGHT))
			  segmentIntersectionsX++;
		      }
		    bool oddIntersectionsX = (segmentIntersectionsX % 2) == 1;
		    insideCurve = oddIntersectionsX;
		  }

		if(dist_method == BRUTE_FORCE)
		  {
		    /* 
		       get the absolute minimum edge distance BRUTE FORCE!!!
		       http://www.lems.brown.edu/vision/courses/medical-imaging-2003/assignments/point-line.html
		    */
		    
		    for(Polygon_2::Edge_const_iterator i = polygon.edges_begin();
			i != polygon.edges_end();
			i++)
		      {
			Point_2 source = i->source(); //x1,y1
			Point_2 target = i->target(); //x2,y2
			Point_2 p(p_x,p_y); //x3,y3
			
			double u = 
			  ((p.x() - source.x())*(target.x() - source.x()) + (p.y() - source.y())*(target.y() - source.y()))/
			  (target - source).squared_length();
			
			Point_2 closest_point_on_edge(source.x() + u*(target.x()-source.x()),
						      source.y() + u*(target.y()-source.y()));
			
			double cur_distance = std::sqrt((closest_point_on_edge - p).squared_length());
			if(cur_distance < distance)
			  distance = cur_distance;
		      }
		  }
		else if(dist_method == K_NEIGHBOR_SEARCH)
		  {
		    // find edges to test against
		    const int NUM_POINTS = 10;
		    Point2DWrapper query(p_x,p_y,0);
		    // search nearest neighbours
		    K_neighbor_search search(tree, query, NUM_POINTS);
		    // cheaper to do redundant tests than to check this for duplicates XD
		    std::vector<Edge2DPtr> closeEdgeSet;
		    for(K_neighbor_search::iterator it = search.begin(); it != search.end(); it++)
		      {
			Point2DWrapper wrapped = it->first;
			// FIXME:
			std::set<Edge2DPtr> edgeSet = wrapped.m_EdgeSet;
			closeEdgeSet.insert(closeEdgeSet.end(), edgeSet.begin(), edgeSet.end());
		      }
		    
		    for(std::vector<Edge2DPtr>::iterator i = closeEdgeSet.begin(); i != closeEdgeSet.end(); i++)
		      {
			Point2D currentPt(p_x,p_y);
			Point2D firstEndPt = *((*i)->getFirst());
			Point2D secondEndPt = *((*i)->getSecond());
			
			double thisDistance = 0;
			
			// determine which range we are in
			bool firstRange = false;
			{
			  Point2D line = secondEndPt - firstEndPt;
			  Point2D a = currentPt - firstEndPt;
			  
			  if(a.dot(line) <= 0)
			    firstRange = true;
			}
			// determine which range we are in
			bool secondRange = false;
			{
			  Point2D line = firstEndPt - secondEndPt;
			  Point2D b = currentPt - secondEndPt;
			  
			  if(b.dot(line) <= 0)
			    secondRange = true;
			}
			
			if(firstRange) {
			  // FIRST PT DIST
			  thisDistance = firstEndPt.distance(currentPt);
			  
			}
			else if(secondRange) {
			  // SECOND PT DIST
			  thisDistance = secondEndPt.distance(currentPt);
			}
			else {
			  // IN MIDDLE RANGE
			  
			  Point2D v1 = firstEndPt - currentPt;
			  Point2D v2 = secondEndPt - currentPt;
			  Point2D v3 = firstEndPt - secondEndPt;
			  
			  double twiceArea = v1.crossMagnitude(v2);
			  thisDistance  = twiceArea / v3.magnitude();
			}
			
			//std::cerr << "thisDistance: " << thisDistance << std::endl;
			
			if(thisDistance < distance)
			  distance = thisDistance;
		      }
		  }
		
		// install in array
		if(!insideCurve)
		  distance = -distance;
		
		if(fabs(distance) < fabs(distanceArray[x][y]))
		  distanceArray[x][y] = distance;
		
		if(distanceArray[x][y] > overallMaxDistance)
		  overallMaxDistance = distanceArray[x][y];
		if(distanceArray[x][y] < overallMinDistance)
		  overallMinDistance = distanceArray[x][y];
	      }
	  }
      }

    timer.stop();
    std::cerr << "\nTook " << timer.time() << " seconds " << "\n";
    std::cerr << "Overall min distance " << overallMinDistance << "\n";
    std::cerr << "Overall max distance " << overallMaxDistance << "\n";

    return distanceArray;
  }
}

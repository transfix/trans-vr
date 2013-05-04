/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Volume Rover.

  Volume Rover is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  Volume Rover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

// MultiContour.h: interface for the MultiContour class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_MULTICONTOUR_H__9DC23294_BACA_494B_AF88_C41B94D54D4F__INCLUDED_)
#define AFX_MULTICONTOUR_H__9DC23294_BACA_494B_AF88_C41B94D54D4F__INCLUDED_

#include <Contouring/ContourExtractor.h>
#include <Contouring/SingleExtractor.h>
#include <Contouring/RGBAExtractor.h>
#include <Contouring/Contour.h>
#include <VolumeWidget/Renderable.h>

struct iPoly;
class Geometry;

///\class MultiContour MultiContour.h
///\brief The MultiContour class is a Renderable instance that can extract and
///	render multiple isocontours from volume data organized in a uniform ijk
///	mesh.
///\author Anthony Thane
///\author John Wiggins
class MultiContour : public Renderable
{
public:
	MultiContour();
	virtual ~MultiContour();

///\fn void setData(unsigned char* data, unsigned int width, unsigned int height, unsigned int depth, double aspectX, double aspectY, double aspectZ, double subMinX, double subMinY, double subMinZ, double subMaxX, double subMaxY, double subMaxZ, double minX, double minY, double minZ, double maxX, double maxY, double maxZ)
///\brief This function is used to load volume data into the class.
///\param data The volume data.
///\param width The width of the volume data
///\param height The height of the volume data
///\param depth The depth of the volume data
///\param aspectX The size of the volume bounding box along the X axis
///\param aspectY The size of the volume bounding box along the Y axis
///\param aspectZ The size of the volume bounding box along the Z axis
///\param subMinX The subvolume extent minimum X coordinate
///\param subMinY The subvolume extent minimum Y coordinate
///\param subMinZ The subvolume extent minimum Z coordinate
///\param subMaxX The subvolume extent maximum X coordinate
///\param subMaxY The subvolume extent maximum Y coordinate
///\param subMaxZ The subvolume extent maximum Z coordinate
///\param minX The extent minimum X coordinate
///\param minY The extent minimum Y coordinate
///\param minZ The extent minimum Z coordinate
///\param maxX The extent maximum X coordinate
///\param maxY The extent maximum Y coordinate
///\param maxZ The extent maximum Z coordinate
	void setData(unsigned char* data, 
		unsigned int width, unsigned int height, unsigned int depth,
		double aspectX, double aspectY, double aspectZ,
		double subMinX, double subMinY, double subMinZ,
		double subMaxX, double subMaxY, double subMaxZ,
		double minX, double minY, double minZ,
		double maxX, double maxY, double maxZ);
///\fn void setData(unsigned char* data, unsigned char* red, unsigned char* green, unsigned char* blue, unsigned int width, unsigned int height, unsigned int depth, double aspectX, double aspectY, double aspectZ, double subMinX, double subMinY, double subMinZ, double subMaxX, double subMaxY, double subMaxZ, double minX, double minY, double minZ, double maxX, double maxY, double maxZ)
///\brief This function is used to load RGBA volume data into the class.
///\param data The volume data
///\param red Red color volume
///\param green Green color volume
///\param blue Blue color volume
///\param width The width of the volume data
///\param height The height of the volume data
///\param depth The depth of the volume data
///\param aspectX The size of the volume bounding box along the X axis
///\param aspectY The size of the volume bounding box along the Y axis
///\param aspectZ The size of the volume bounding box along the Z axis
///\param subMinX The subvolume extent minimum X coordinate
///\param subMinY The subvolume extent minimum Y coordinate
///\param subMinZ The subvolume extent minimum Z coordinate
///\param subMaxX The subvolume extent maximum X coordinate
///\param subMaxY The subvolume extent maximum Y coordinate
///\param subMaxZ The subvolume extent maximum Z coordinate
///\param minX The extent minimum X coordinate
///\param minY The extent minimum Y coordinate
///\param minZ The extent minimum Z coordinate
///\param maxX The extent maximum X coordinate
///\param maxY The extent maximum Y coordinate
///\param maxZ The extent maximum Z coordinate
	void setData(unsigned char* data, unsigned char* red, unsigned char* green, unsigned char* blue,
		unsigned int width, unsigned int height, unsigned int depth,
		double aspectX, double aspectY, double aspectZ,
		double subMinX, double subMinY, double subMinZ,
		double subMaxX, double subMaxY, double subMaxZ,
		double minX, double minY, double minZ,
		double maxX, double maxY, double maxZ);

///\fn void addContour(int ID, float isovalue, float R, float G, float B)
///\brief This function adds a contour to be extracted/rendered.
///\param ID A unique identifier for the contour
///\param isovalue The isovalue of the contour
///\param R The red component of the contour's color
///\param G The green component of the contour's color
///\param B The blue component of the contour's color
	void addContour(int ID, float isovalue, float R, float G, float B);
///\fn void removeContour(int ID)
///\brief This function removes a previously added contour.
///\param ID The unique identifier of the contour to be removed
	void removeContour(int ID);
///\fn void removeAll()
///\brief This function removes all contours.
	void removeAll();
///\fn void setIsovalue(int ID, float isovalue)
///\brief This function changes the isovalue for a specific contour.
///\param ID The unique identifier of the contour to be modified
///\param isovalue The new isovalue
	void setIsovalue(int ID, float isovalue);
///\fn void setColor(int ID, float R, float G, float B)
///\brief This function changes the color of a specific contour.
///\param ID The unique identifier of the contour to be modified
///\param R The red component of the new color
///\param G The green component of the new color
///\param B The blue component of the new color
	void setColor(int ID, float R, float G, float B);
	virtual void setWireframeMode(bool state);
	virtual void setSurfWithWire(bool state);

///\fn void renderContours() const
///\brief This function renders all the contours. render() should still be
///	used, per the Renderable interface.
	void renderContours() const;
///\fn void forceExtraction() const
///\brief This function forces the extraction of isocontours. Normally,
///	isocontours are only extracted when their isovalue changes.
	void forceExtraction() const;
	virtual bool render();

	//iPoly* getIPoly();
///\fn Geometry* getGeometry()
///\brief This function returns a Geometry object containing all the contours.
///\return A pointer to a Geometry object
	Geometry* getGeometry();

///\fn int getNumVerts()
///\brief This function returns the number of vertices in all the contours.
///\return The number of vertices
	int getNumVerts();
///\fn int getNumTris()
///\brief This function returns the number of triangles in all the contours.
///\return The number of triangles
	int getNumTris();

private:
///\class MultiContour::MultiContourNode
///\brief This private class manages the Contour objects contained within a
///	MultiContour instance.
	class MultiContourNode
	{
	public:
	  MultiContourNode(int ID, float isovalue, float R, float G, float B, bool wire, bool surfwire, MultiContourNode* next) : m_Next(next) 
		{
			m_Contour.setID(ID);
			m_Contour.setIsovalue(isovalue);
			m_Contour.setWireframeMode(wire);
			m_Contour.setSurfWithWire(surfwire);
			m_Contour.setSingleColor(R,G,B, true);
		};
		virtual ~MultiContourNode() 
		{ 
			delete m_Next; 
		};
		MultiContourNode* m_Next;
		Contour m_Contour;
	};

protected:
	void setDefaults();

	void resetContours();

	SingleExtractor m_SingleExtractor;
	RGBAExtractor m_RGBAExtractor;
	ContourExtractor* m_ContourExtractor;

	MultiContourNode* m_Head;

	bool m_DataLoaded;
	bool m_WireframeRender;
	bool m_SurfWithWire;

	// matrix used to scale and center the contour into
	// the right place
	Matrix* m_Matrix;
	// matrix used to align the data to the correct space
	// for saving
	Matrix* m_SaveMatrix;



};

#endif // !defined(AFX_MULTICONTOUR_H__9DC23294_BACA_494B_AF88_C41B94D54D4F__INCLUDED_)

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

// TransferArray.h: interface for the TransferArray class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_TRANSFERARRAY_H__F164CC87_82B8_408F_B5EF_1B1FCA329E20__INCLUDED_)
#define AFX_TRANSFERARRAY_H__F164CC87_82B8_408F_B5EF_1B1FCA329E20__INCLUDED_

class ColorTableInformation;

///\ingroup libRenderServer
///\class TransferArray TransferArray.h
///\brief The TransferArray class is a simplified version of
///	ColorTableInformation that provides a more concise description of a transfer
///	function.
///\author Anthony Thane
class TransferArray  
{
public:
	///\ingroup libRenderServer
	///\struct TransferElement
	///\brief A TransferElement represents a single node in a TransferArray.
	struct TransferElement {
		TransferElement() : m_Position(1.0),
			m_Alpha(0.0), m_Red(0.0), m_Green(0.0), m_Blue(0.0) {}
		TransferElement(double position, double alpha, double red, double green, double blue) : 
			m_Position(position), m_Alpha(alpha), m_Red(red), m_Green(green), m_Blue(blue) {}
		double m_Position;
		double m_Alpha;
		double m_Red;
		double m_Green;
		double m_Blue;
	};

	TransferArray(unsigned int arrayGuess = 16);
	virtual ~TransferArray();

	void buildFromColorTable(const ColorTableInformation& colorTableInformation);
	void addElement(const TransferElement& transferElement);
	void addElement(double position, double alpha, double red, double green, double blue);

	unsigned int getNumElements() const;

	TransferElement& operator[](unsigned int index);
	const TransferElement& operator[](unsigned int index) const;

protected:
	void setDefaults();
	void destroyArray();
	void allocateArray(unsigned int arraySize);
	void doubleArray();
	
	TransferElement* m_Array;
	unsigned int m_ArraySize;
	unsigned int m_NumElements;


};

#endif // !defined(AFX_TRANSFERARRAY_H__F164CC87_82B8_408F_B5EF_1B1FCA329E20__INCLUDED_)

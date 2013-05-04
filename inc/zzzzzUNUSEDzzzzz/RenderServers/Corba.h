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

// Corba.h: interface for the Corba class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_CORBA_H__FEA5B814_E97B_41EF_890E_98E5AE054813__INCLUDED_)
#define AFX_CORBA_H__FEA5B814_E97B_41EF_890E_98E5AE054813__INCLUDED_

//#include <OB/CORBA.h>
namespace CORBA {
	class ORB;
}

///\ingroup libRenderServer
///\class Corba Corba.h
///\brief This class handles initialization of Corba for RenderServer
///	instances.
///\author Anthony Thane
class Corba  
{
public:
	virtual ~Corba();

	static bool initCorba();
	static CORBA::ORB* getOrb();

protected:
	Corba();
	void setDefaults();
	bool initCorbaPrivate();
	CORBA::ORB* getOrbPrivate();

	static Corba ms_Corba;

	CORBA::ORB* m_Orb;
	bool m_Initialized;

};

#endif // !defined(AFX_CORBA_H__FEA5B814_E97B_41EF_890E_98E5AE054813__INCLUDED_)

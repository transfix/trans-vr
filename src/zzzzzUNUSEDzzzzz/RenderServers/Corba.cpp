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

// Corba.cpp: implementation of the Corba class.
//
//////////////////////////////////////////////////////////////////////

#include <RenderServers/Corba.h>
#include <OB/CORBA.h>

Corba Corba::ms_Corba;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Corba::Corba()
{
	setDefaults();
	/*	int i;


	if (!initCorbaPrivate()) {
		//qDebug("Corba initialization failed!");
		i = 0;
	}
	else {
		i = 1;
	}
	*/
}

Corba::~Corba()
{
	/*if (m_Orb) {
		m_Orb->destroy();
		delete m_Orb;
		m_Orb = 0;
	}*/
}

void Corba::setDefaults()
{
	m_Orb = 0;
	m_Initialized = false;
}

bool Corba::initCorba()
{
	return ms_Corba.initCorbaPrivate();
}

bool Corba::initCorbaPrivate()
{
	char name[] = "RenderServer";
	char option[] = "-OAthreaded";
	char * argv1[] = {&(name[0]), &(option[0])};
	char** argv = &(argv1[0]);
	int argc = 2;

	try {
		m_Orb = CORBA::ORB_init(argc, argv);
	}
	catch(CORBA::SystemException&) {
		m_Initialized = false;
		return false;
	}
	m_Initialized = true;
	return true;
}

CORBA::ORB* Corba::getOrb()
{
	return ms_Corba.getOrbPrivate();
}

CORBA::ORB* Corba::getOrbPrivate()
{
	if (!m_Initialized) {
		if (!initCorbaPrivate()) {
			return 0;
		}
		else {
			return m_Orb;
		}
	}
	else {
		return m_Orb;
	}
}


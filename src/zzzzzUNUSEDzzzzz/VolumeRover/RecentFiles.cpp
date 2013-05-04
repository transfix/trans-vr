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

// RecentFiles.cpp: implementation of the RecentFiles class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeRover/RecentFiles.h>
#include <qfileinfo.h>
#include <qsettings.h>
//Added by qt3to4:
#include <Q3PopupMenu>
//#include <crtdbg.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

RecentFiles::RecentFiles(QMenu* menu, const QObject* receiver, const char* member, int numFiles) :
m_Menu(menu), m_NumSlots(numFiles), m_Receiver(receiver), m_Member(member)
{
	m_Files = new QString[numFiles];
	m_Indexes = new int[numFiles+1];
	m_NumActual = 0;
	getFromSettings();
}

RecentFiles::~RecentFiles()
{
	updateSettings();
	removeFromMenu();
	delete [] m_Files;
	delete [] m_Indexes;
}

void RecentFiles::changeNumFiles(int numFiles)
{
	if (m_NumSlots!=numFiles) {
		removeFromMenu();
		QString* newFiles = new QString[numFiles];

		delete [] m_Indexes;
		m_Indexes = new int[numFiles+1];

		// copy over the old files
		int c;
		for (c=0; c<m_NumSlots && c<numFiles; c++) {
			newFiles[c] = m_Files[c];
		}

		delete [] m_Files;
		m_Files = newFiles;

		m_NumSlots = numFiles;
		if (m_NumActual>numFiles) 
			m_NumActual=numFiles;

		addToMenu();
	}
}

void RecentFiles::enable()
{
	addToMenu();
}

void RecentFiles::disable()
{
	removeFromMenu();
}

void RecentFiles::updateRecentFiles(const QString& filename)
{
	// check for freakish case
	if (m_NumSlots<=0) return;

	removeFromMenu();

	/*QFileInfo info(filename);
	QString newFile(info.fileName());*/

	// if the name already exists, we will promote it to the top
	int spot = findName(filename);
	int actualSpot = (spot<0?m_NumActual:spot);
	
	int c;
	for (c=actualSpot; c>0; c--) {
		if (c<m_NumSlots)
			m_Files[c] = m_Files[c-1];
	}
	m_Files[0] = filename;

	// update the number of actual file names
	if (spot<0) {
		m_NumActual++;
		if (m_NumActual>m_NumSlots) {
			m_NumActual = m_NumSlots;
		}
	}

	addToMenu();
	updateSettings();

	   /* Check heap status */
	//qDebug("About to check heap status");
	//_CrtCheckMemory();
}

QString RecentFiles::getFileName(int whichFile) const
{
	return m_Files[whichFile];
}

int RecentFiles::findName(const QString& fileName) const
{
	QFileInfo fileInfo1(fileName), fileInfo2;
	int c;
	for (c=0; c<m_NumActual; c++) {
		fileInfo2.setFile(m_Files[c]);
		if (fileInfo1.fileName()==fileInfo2.fileName()) {
			return c;
		}
	}
	return -1;
}

void RecentFiles::getFromSettings()
{
	QSettings settings;

	settings.insertSearchPath(QSettings::Windows, "/CCV");

	int numrecent = settings.readNumEntry("/Volume Rover/Recent Files/NumActual", 0);
	int numslots = settings.readNumEntry("/Volume Rover/Recent Files/NumSlots", m_NumSlots);
	if (numrecent==0) {
		//m_Files[0] = "blah";
		//m_Files[1] = "lahal";
		m_NumActual = 0;

	}
	else {
		changeNumFiles(numslots);
		int c;
		QString key;
		for (c=0; c<m_NumActual; c++) {
			//key.sprintf("Volume Rover/Recent Files/Files/%03d"
		}
		m_NumActual = 0;
	}


	
}

void RecentFiles::updateSettings()
{
	QSettings settings;

	settings.insertSearchPath(QSettings::Windows, "/CCV");

	settings.writeEntry("/Volume Rover/Recent Files/", m_NumActual);
	settings.writeEntry("/Volume Rover/Recent Files/NumSlots", m_NumSlots);


}

void RecentFiles::addToMenu()
{
	QFileInfo fileInfo;
	if (m_NumActual>0) {
		int c;
		for (c=m_NumActual-1; c>=0; c--) {
			fileInfo.setFile(m_Files[c]);
			m_Indexes[c] = m_Menu->insertItem(fileInfo.fileName(), m_Receiver, m_Member, 0, -1, 5);
			m_Menu->setItemParameter(m_Indexes[c], c);
		}
		m_Indexes[m_NumActual] = m_Menu->insertSeparator(5);
	}
}

void RecentFiles::removeFromMenu()
{
	int c;
	if (m_NumActual>0) {
		for (c=0; c<=m_NumActual; c++) {
			m_Menu->removeItem(m_Indexes[c]);
		}
	}
}


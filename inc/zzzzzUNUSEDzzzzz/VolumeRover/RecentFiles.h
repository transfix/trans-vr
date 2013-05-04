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

// RecentFiles.h: interface for the RecentFiles class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_RECENTFILES_H__FB4D271B_C7BF_4AF7_B23D_6AC204F95026__INCLUDED_)
#define AFX_RECENTFILES_H__FB4D271B_C7BF_4AF7_B23D_6AC204F95026__INCLUDED_

#include <qobject.h>
#include <q3popupmenu.h>
#include <qstring.h>

class RecentFiles: public QObject
{
public:
	RecentFiles(QMenu* menu, const QObject* receiever, const char* member, int numFiles = 4);
	virtual ~RecentFiles();

	void changeNumFiles(int numFiles);
	void enable();
	void disable();
	void updateRecentFiles(const QString& filename);
	QString getFileName(int whichFile) const;

protected:
	int findName(const QString& fileName) const;
	void getFromSettings();
	void updateSettings();

	void addToMenu();
	void removeFromMenu();

	QMenu* m_Menu;
	int m_NumSlots;
	int m_NumActual;

	QString* m_Files;
	int* m_Indexes;

	const QObject* m_Receiver;
	const char* m_Member;

	bool m_InMenu;



};

#endif // !defined(AFX_RECENTFILES_H__FB4D271B_C7BF_4AF7_B23D_6AC204F95026__INCLUDED_)

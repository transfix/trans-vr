/*
  Copyright 2006-2008 The University of Texas at Austin

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

#ifndef __POINTCLASSFILE_H__
#define __POINTCLASSFILE_H__

#include <qstring.h>
#include <qxml.h>

class VolumeGridRover;

/* The content handler for reading a point class file */
class ContourFileContentHandler : public QXmlDefaultHandler
{
 public:
  ContourFileContentHandler(VolumeGridRover *vgr);
  virtual void setDocumentLocator(QXmlLocator *locator);
  //virtual bool startDocument();
  virtual bool endDocument();
  //virtual bool startPrefixMapping(const QString & prefix, const QString & uri );
  //virtual bool endPrefixMapping(const QString & prefix);
  virtual bool startElement(const QString & namespaceURI, const QString & localName, const QString & qName, const QXmlAttributes & atts);
  virtual bool endElement(const QString & namespaceURI, const QString & localName, const QString & qName);
  virtual bool characters (const QString & ch);
  //virtual bool ignorableWhitespace(const QString & ch);
  //virtual bool processingInstruction(const QString & target, const QString & data);
  //virtual bool skippedEntity(const QString & name);
  virtual QString errorString();
 private:

  VolumeGridRover *m_VolumeGridRover;
  QXmlLocator *m_Locator;
};

#endif

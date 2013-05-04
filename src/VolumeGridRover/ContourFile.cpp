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

#include <VolumeGridRover/ContourFile.h>
#include <VolumeGridRover/VolumeGridRover.h>
#include <CVC/App.h>
#include <boost/format.hpp>

ContourFileContentHandler::ContourFileContentHandler(VolumeGridRover *vgr) 
  : m_VolumeGridRover(vgr) {}
void ContourFileContentHandler::setDocumentLocator(QXmlLocator *locator) { m_Locator = locator; }
//bool ContourFileContentHandler::startDocument() { return true; }
bool ContourFileContentHandler::endDocument() { /*if(m_CurrentPointClass) delete m_CurrentPointClass;*/ return true; }
//bool ContourFileContentHandler::startPrefixMapping(const QString &prefix, const QString &uri) { return true; }
//bool ContourFileContentHandler::endPrefixMapping(const QString &prefix) { return true; }
bool ContourFileContentHandler::startElement(const QString &namespaceURI, const QString &localName, const QString &qName, const QXmlAttributes &atts)
{
  if(!namespaceURI.isNull()) cvcapp.log(5, boost::str(boost::format("Using namespace: %s")%namespaceURI.ascii()));
  cvcapp.log(5, boost::str(boost::format("Qualified name: %s")%qName.ascii()));

  //if(localName == "contours");

  return true;
}
bool ContourFileContentHandler::endElement(const QString &namespaceURI, const QString &localName, const QString &qName)
{
  return true;
}
bool ContourFileContentHandler::characters(const QString &ch)
{
  return false;
}
//bool ContourFileContentHandler::ignorableWhitespace(const QString &ch) { return true; }
//bool ContourFileContentHandler::processingInstruction(const QString &target, const QString &data) { return true; }
//bool ContourFileContentHandler::skippedEntity(const QString &name) { return true; }
QString ContourFileContentHandler::errorString()
{ 
  return QString("Error");
}

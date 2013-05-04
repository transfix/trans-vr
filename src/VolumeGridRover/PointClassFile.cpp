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

#include <VolumeGridRover/VolumeGridRover.h>
#include <VolumeGridRover/PointClassFile.h>
#include <ui_VolumeGridRoverBase.h>
#include <CVC/App.h>

/************* Point Class XML File reading handler ************************/

PointClassFileContentHandler::PointClassFileContentHandler(VolumeGridRover *vgr) 
  : m_CurrentError(NoError), m_VolumeGridRover(vgr), m_CurrentPointClass(NULL), m_InsidePoint(false), m_InsidePointClassDoc(false) {}
void PointClassFileContentHandler::setDocumentLocator(QXmlLocator *locator) { m_Locator = locator; }
//bool PointClassFileContentHandler::startDocument() { return true; }
bool PointClassFileContentHandler::endDocument() { /*if(m_CurrentPointClass) delete m_CurrentPointClass;*/ return true; }
//bool PointClassFileContentHandler::startPrefixMapping(const QString &prefix, const QString &uri) { return true; }
//bool PointClassFileContentHandler::endPrefixMapping(const QString &prefix) { return true; }
bool PointClassFileContentHandler::startElement(const QString &namespaceURI, const QString &localName, const QString &qName, const QXmlAttributes &atts)
{
  if(!namespaceURI.isNull()) cvcapp.log(5, boost::str(boost::format("Using namespace: %s")%namespaceURI.ascii()));
  cvcapp.log(5, boost::str(boost::format("Qualified name: %s")%qName.ascii()));

  if(localName == "pointclassdoc")
    {
      m_InsidePointClassDoc = true;
    }
  else if(localName == "pointclass")
    {
      if(!m_InsidePointClassDoc)
	{
	  m_CurrentError = NotInsidePointClassDoc;
	  return false;
	}
	
      if(atts.index("name")==-1)
	{
	  m_CurrentError = MissingNameAttribute;
	  return false;
	}
      if(atts.index("color")==-1)
	{
	  m_CurrentError = MissingColorAttribute;
	  return false;
	}
      if(atts.index("variable")==-1)
	{
	  m_CurrentError = MissingVariableAttribute;
	  return false;
	}
      if(atts.index("timestep")==-1)
	{
	  m_CurrentError = MissingTimestepAttribute;
	  return false;
	}
		
      QString classname = atts.value("name");
      QColor classcolor(atts.value("color"));
      int variable = atts.value("variable").toInt();
      int timestep = atts.value("timestep").toInt();
		
      if(!classcolor.isValid())
	{
	  m_CurrentError = InvalidColor;
	  return false;
	}
		
      if(m_CurrentPointClass != NULL)
	{
	  m_CurrentError = IllegalPointClassDeclaration;
	  return false;
	}
		
      if(variable < 0 && variable >= int(m_VolumeGridRover->m_VolumeFileInfo.numVariables()))
	{
	  m_CurrentError = VariableOutOfBounds;
	  return false;
	}
		
      if(timestep < 0 && timestep >= int(m_VolumeGridRover->m_VolumeFileInfo.numTimesteps()))
	{
	  m_CurrentError = TimeStepOutOfBounds;
	  return false;
	}
		
      m_VolumeGridRover->m_PointClassList[variable][timestep]->append(m_CurrentPointClass = new PointClass(classcolor,classname));
      if(m_VolumeGridRover->_ui->m_Variable->currentItem() == variable && m_VolumeGridRover->_ui->m_Timestep->value() == timestep)
	m_VolumeGridRover->_ui->m_PointClass->insertItem(classname);
      cvcapp.log(5, boost::str(boost::format("PointClassFileContentHandler::startElement(): var: %d, time: %d, classcolor: %s, classname: %s")%variable%timestep%classcolor.name().ascii()%classname.ascii()));
    }
  else if(localName == "point")
    {
      if(!m_InsidePointClassDoc)
	{
	  m_CurrentError = NotInsidePointClassDoc;
	  return false;
	}
	
      m_InsidePoint = true;
    }
	
  return true;
}
bool PointClassFileContentHandler::endElement(const QString &namespaceURI, const QString &localName, const QString &qName)
{
  if(!namespaceURI.isNull()) cvcapp.log(5, boost::str(boost::format("Using namespace: %s")%namespaceURI.ascii()));
  cvcapp.log(5, boost::str(boost::format("Qualified name: %s")%qName.ascii()));

  if(localName == "pointclassdoc")
    {
      m_InsidePointClassDoc = false;
    }
  if(localName == "pointclass")
    {
      m_CurrentPointClass = NULL;
    }
  else if(localName == "point")
    {
      m_InsidePoint = false;
    }
 
  return true;
}
bool PointClassFileContentHandler::characters(const QString &ch)
{
  if((m_CurrentPointClass != NULL) && m_InsidePoint)
    {
      QStringList point(QStringList::split(" ",ch));
      bool ok;
      unsigned int x,y,z;
		
      x = point[0].toUInt(&ok);
      if(!ok)
	{
	  m_CurrentError = InvalidXParameter;
	  return false;
	}
		
      y = point[1].toUInt(&ok);
      if(!ok)
	{
	  m_CurrentError = InvalidYParameter;
	  return false;
	}
		
      z = point[2].toUInt(&ok);
      if(!ok)
	{
	  m_CurrentError = InvalidZParameter;
	  return false;
	}
		
      if(x >= m_VolumeGridRover->m_VolumeFileInfo.XDim())
	{
	  m_CurrentError = XParameterOutOfBounds;
	  return false;
	}
		
      if(y >= m_VolumeGridRover->m_VolumeFileInfo.YDim())
	{
	  m_CurrentError = YParameterOutOfBounds;
	  return false;
	}

      if(z >= m_VolumeGridRover->m_VolumeFileInfo.ZDim())
	{
	  m_CurrentError = ZParameterOutOfBounds;
	  return false;
	}
      cvcapp.log(5, boost::str(boost::format("PointClassFileContentHandler::characters(): Adding point to class %s: (%d,%d,%d)")%
				  m_CurrentPointClass->getName().ascii()%x%y%z));
      m_CurrentPointClass->addPoint(x,y,z);		
    }
	
  return true; 
}
//bool PointClassFileContentHandler::ignorableWhitespace(const QString &ch) { return true; }
//bool PointClassFileContentHandler::processingInstruction(const QString &target, const QString &data) { return true; }
//bool PointClassFileContentHandler::skippedEntity(const QString &name) { return true; }
QString PointClassFileContentHandler::errorString()
{ 
  const char *errors[] = { "No Error", "Root element is not <pointclassdoc>!", "Missing name attribute", "Missing color attribute", "Missing variable attribute",
			   "Missing timestep attribute", "Invalid color", "Illegal point class declaration!", "Variable out of bounds",
			   "Timestep out of bounds", "Invalid X parameter", "Invalid Y parameter", "Invalid Z parameter",
			   "X parameter out of bounds", "Y parameter out of bounds", "Z parameter out of bounds" };
  return errors[m_CurrentError];
}

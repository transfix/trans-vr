/*
  Copyright 2002-2003 The University of Texas at Austin

	Authors: Anthony Thane <thanea@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeLibrary.

  VolumeLibrary is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeLibrary is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

// VolumeRendererFactory.h: interface for the VolumeRendererFactory class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_VOLUMERENDERERFACTORY_H__565249A1_B920_4AFE_8A52_D1099907333C__INCLUDED_)
#define AFX_VOLUMERENDERERFACTORY_H__565249A1_B920_4AFE_8A52_D1099907333C__INCLUDED_

namespace OpenGLVolumeRendering {
	class RGBABase;
	class UnshadedBase;

	class VolumeRendererFactory  
	{
	public:
		VolumeRendererFactory();
		virtual ~VolumeRendererFactory();

		static RGBABase* getRGBARenderer();

		static UnshadedBase* getUnshadedRenderer();

	};

};

#endif // !defined(AFX_VOLUMERENDERERFACTORY_H__565249A1_B920_4AFE_8A52_D1099907333C__INCLUDED_)

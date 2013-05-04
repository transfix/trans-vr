/*
  Copyright 2010-2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of libCVC.

  libCVC is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  libCVC is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: HDF5_Utilities.h 5881 2012-07-20 19:34:04Z edwardsj $ */

#ifndef __CVC_HDF5_UTILITIES_H__
#define __CVC_HDF5_UTILITIES_H__

#include <CVC/Exception.h>
#include <CVC/Types.h>
#include <CVC/Dimension.h>
#include <CVC/BoundingBox.h>
#include <CVC/Namespace.h>

#include <CVC/App.h>

#if defined (WIN32)
#include <cpp/H5Cpp.h> //it appears this has changed back to H5Cpp.h - transfix 03/30/2012
// but not on windows...
#else 
#include <H5Cpp.h>
#endif

#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/scoped_array.hpp>
#include <boost/algorithm/minmax_element.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/format.hpp>

#include <string>
#include <list>
#include <iostream>
#include <cstring>

/*
 * CVC hdf5 schema -- Joe R. -- 01/04/2010
 *  / - root
 *  |_ cvc/ - root cvc data hierarchy
 *     |_ geometry/ - placeholder, define fully later
 *     |_ transfer_functions/ - placeholder, define fully later
 *     |_ volumes/
 *        |_ <volume name> - volume group containing image or volume data
 *           |               Default volume name is 'volume'.  This group has the following attribs:
 *           |               - VolMagick_version (uint64)
 *           |               - XMin, YMin, ZMin,
 *           |                 XMax, YMax, ZMax  (double) - bounding box
 *           |               - XDim, YDim, ZDim (uint64) - volume dimensions
 *           |               - voxelTypes (uint64 array) - the type of each variable
 *           |               - numVariables (uint64)
 *           |               - numTimesteps (uint64)
 *           |               - min_time (double)
 *           |               - max_time (double)
 *           |_ <volume name>:<variable (int)>:<timestep (int)> - dataset for a volume.  each variable
 *                                                                of each timestep has it's own dataset.
 *                                                                Each volume dataset has the following
 *                                                                attributes:
 *                                                                - min (double) - min voxel value
 *                                                                - max (double) - max voxel value
 *                                                                - name (string) - variable name
 *                                                                - voxelType (uint64) - type of this dataset
 *  
 */

namespace CVC_NAMESPACE
{
  CVC_DEF_EXCEPTION(InvalidHDF5File);
  CVC_DEF_EXCEPTION(HDF5Exception);

  namespace HDF5_Utilities
  {
    const hsize_t ATTRIBUTE_STRING_MAXLEN = 256;

    // --------------------
    // getPredType
    // --------------------
    // Purpose:
    //   Returns a PredType given a CVC::DataType.
    // ---- Change History ----
    // 12/31/2009 -- Joe R. -- Initial implementation.    
    H5::PredType getPredType(DataType vt);

    // ------------------
    // getH5File
    // ------------------
    // Purpose:
    //   Gets an H5File object for the provided filename, either creating or
    //   opening a file to do so.
    // ---- Change History ----
    // 12/29/2009 -- Joe R. -- Initial implementation.
    // 08/27/2010 -- Joe R. -- Don't create by default.
    boost::shared_ptr<H5::H5File> getH5File(const std::string& filename,
                                            bool create = false);

    // -----------------
    // getGroup
    // -----------------
    // Purpose:
    //   Gets a Group object for the provided file and object path, either creating or
    //   opening groups along the way to do so.
    // ---- Change History ----
    // 12/29/2009 -- Joe R. -- Initial implementation.    
    boost::shared_ptr<H5::Group> getGroup(const H5::H5File& file,
                                          const std::string& groupPath,
                                          bool create = true);
    // -----------------
    // getDataSet
    // -----------------
    // Purpose:
    //   Gets a DataSet object for the provided file and object path, creating
    //   groups along the way if create == true
    // ---- Change History ----
    // 06/17/2011 -- Joe R. -- Initial implementation.    
    boost::shared_ptr<H5::DataSet> getDataSet(const H5::H5File& file,
                                              const std::string& dataSetPath,
                                              bool create = true);

    // -----------------
    // unlink
    // -----------------
    // Purpose:
    //  Unlinks the object at the specified path.
    // ---- Change History ----
    // 07/15/2011 -- Joe R. -- Initial implementation.    
    void unlink(const H5::H5File& file,
                const std::string& objectPath);

    // ---------------------
    // hasAttribute
    // ---------------------
    // Purpose:
    //   Returns true if object has named attribute.
    // ---- Change History ----
    // 12/29/2009 -- Joe R. -- Initial implementation.    
    bool hasAttribute(const H5::H5Object& obj,
                      const std::string& name);                 

    // ---------------------
    // getAttribute
    // ---------------------
    // Purpose:
    //   Reads a 1D array of T elements from an
    //   attribute of an object.
    // ---- Change History ----
    // 12/29/2009 -- Joe R. -- Initial implementation.
    // 06/24/2011 -- Joe R. -- Changed to template.
    template<class T>
    inline void getAttribute(const H5::H5Object &obj,
                             const std::string& name,
                             size_t len,
                             T* values)
    {
      using namespace H5;

      Attribute attr;
      attr = obj.openAttribute(name);
    
      //Check the dataspace of the attribute to make sure it's the same size
      //as what we need.
      DataSpace attrDS = attr.getSpace();
      int num_dims = attrDS.getSimpleExtentNdims();
      if(num_dims == 1)
        {
          hsize_t dims[1];
          attrDS.getSimpleExtentDims(dims);
          if(dims[0] != len)
            throw AttributeIException(BOOST_CURRENT_FUNCTION,
                                      "Attribute length mismatch");
        }
      else
        {
          throw AttributeIException(BOOST_CURRENT_FUNCTION,
                                    "Invalid number of dimensions (expecting 1D)");
        }
    
      attr.read(getPredType(cvcapp.dataType<T>()), values);
    }

    // ---------------------
    // getAttribute
    // ---------------------
    // Purpose:
    //   Reads an attribute of type T with specified 
    //   name from the specified object.
    // ---- Change History ----
    // 12/29/2009 -- Joe R. -- Initial implementation.
    // 06/24/2011 -- Joe R. -- Changed to template.
    template<class T>
    inline void getAttribute(const H5::H5Object& obj,
                             const std::string& name,
                             T &value)
    {
      getAttribute(obj, name, 1, &value);
    }

    // ---------------------
    // getAttribute
    // ---------------------
    // Purpose:
    //   Gets a string attribute from an object.
    // ---- Change History ----
    // 12/29/2009 -- Joe R. -- Initial implementation.
    // 06/24/2011 -- Joe R. -- Changed to specialized template
    template<>
    inline void getAttribute<std::string>(const H5::H5Object& obj,
                                          const std::string& name,
                                          std::string& value)
    {
      using namespace H5;
      Attribute attr = obj.openAttribute(name);
      //Not satisfied with the below hack, TODO: investigate later
      char cvalue[ATTRIBUTE_STRING_MAXLEN+1];
      memset(cvalue,0,sizeof(char)*(ATTRIBUTE_STRING_MAXLEN+1));
      attr.read(StrType(0,ATTRIBUTE_STRING_MAXLEN), cvalue);
      value = std::string(cvalue);
    }

    // ---------------------
    // setAttribute
    // ---------------------
    // Purpose:
    //   Writes a 1D array of T elements to an
    //   attribute of an object.
    // ---- Change History ----
    // 12/29/2009 -- Joe R. -- Initial implementation.
    // 06/24/2011 -- Joe R. -- Changed to template.
    template<class T>
    inline void setAttribute(const H5::H5Object& obj,
                             const std::string& name,
                             size_t len,
                             const T* values)
    {
      using namespace H5;

      Attribute attr;
      try
        {
          attr = obj.openAttribute(name);

          //Check the dataspace of the attribute to make sure it's the same size
          //as what we need.
          DataSpace attrDS = attr.getSpace();
          int num_dims = attrDS.getSimpleExtentNdims();
          if(num_dims == 1)
            {
              hsize_t dims[1];
              attrDS.getSimpleExtentDims(dims);
              if(dims[0] != len)
                {
                  obj.removeAttr(name);
                  throw AttributeIException(BOOST_CURRENT_FUNCTION);
                }
            }
          else
            {
              obj.removeAttr(name);
              throw AttributeIException(BOOST_CURRENT_FUNCTION);
            }
        }
      catch(AttributeIException& e)
        {
          hsize_t dim[] = { len };
          DataSpace attrDS(1,dim);
          attr = obj.createAttribute(name,
                                     getPredType(cvcapp.dataType<T>()),
                                     attrDS);
        }

      attr.write(getPredType(cvcapp.dataType<T>()), values);
    }

    // ---------------------
    // setAttribute
    // ---------------------
    // Purpose:
    //   Forcing DataType enum to use uint64 type
    // ---- Change History ----
    // 09/02/2011 -- Joe R. -- Initial implementation.
    template<>
    inline void setAttribute<DataType>(const H5::H5Object& obj,
                                       const std::string& name,
                                       size_t len,
                                       const DataType* values)
    {
      std::vector<uint64> dt_values(len);
      for(size_t i = 0; i < len; i++)
        dt_values[i] = uint64(values[i]);
      setAttribute(obj, name, len, &(dt_values[0]));
    }

    // ---------------------
    // setAttribute
    // ---------------------
    // Purpose:
    //   Writes an attribute of type T with specified 
    //   name to the specified object.
    // ---- Change History ----
    // 12/29/2009 -- Joe R. -- Initial implementation.
    // 06/24/2011 -- Joe R. -- Changed to template.
    template<class T>
    inline void setAttribute(const H5::H5Object& obj,
                             const std::string& name,
                             const T &value)
    {
      setAttribute(obj, name, 1, &value);
    }

    // ---------------------
    // setAttribute
    // ---------------------
    // Purpose:
    //   Sets a string attribute on an object.
    // ---- Change History ----
    // 12/29/2009 -- Joe R. -- Initial implementation.
    // 06/24/2011 -- Joe R. -- Changed to specialized template
    template<>
    inline void setAttribute<std::string>(const H5::H5Object& obj,
                                          const std::string& name,
                                          const std::string& value)
    {
      using namespace H5;

      Attribute attr;
      try
        {
          obj.removeAttr(name);
        }
      catch(H5::Exception& e)
        {}

      hsize_t dim[] = { 1 };
      DataSpace attrDS(1,dim);
      attr = obj.createAttribute(name,
                                 StrType(0,ATTRIBUTE_STRING_MAXLEN),
                                 attrDS);

      attr.write(StrType(0,ATTRIBUTE_STRING_MAXLEN), value.c_str());
    }

    // ---------------------
    // setAttribute
    // ---------------------
    // Purpose:
    //   Sets a string attribute on an object via a char*
    // ---- Change History ----
    // 08/28/2011 -- Joe R. -- Initial implementation.
    inline void setAttribute(const H5::H5Object& obj,
                             const std::string& name,
                             const char* value)
    {
      setAttribute(obj,name,std::string(value));
    }

    // ---------------------
    // isGroup
    // ---------------------
    // Purpose:
    //   testing for group in HDF5 file
    // ---- Change History ----
    // 06/24/2011 -- Joe R. -- Initial implementation.
    bool isGroup(const std::string& hdf5_filename,
                 const std::string& hdf5_objname);

    // ---------------------
    // isDataSet
    // ---------------------
    // Purpose:
    //   testing for dataset in HDF5 file
    // ---- Change History ----
    // 06/24/2011 -- Joe R. -- Initial implementation.
    bool isDataSet(const std::string& hdf5_filename,
                   const std::string& hdf5_objname);

    // ---------------------
    // objectExists
    // ---------------------
    // Purpose:
    //   Returns true of the specified object is a
    //   dataset or group in the specified hdf5 file.
    // ---- Change History ----
    // 07/15/2011 -- Joe R. -- Initial implementation.
    bool objectExists(const std::string& hdf5_filename,
                      const std::string& hdf5_objname);

    // ---------------------
    // removeObject
    // ---------------------
    // Purpose:
    //   Removes the specified object from the HDF5
    //   file.
    // ---- Change History ----
    // 07/15/2011 -- Joe R. -- Initial implementation.
    void removeObject(const std::string& hdf5_filename,
                      const std::string& hdf5_objname);

    // --------------
    // createHDF5File
    // --------------
    // Purpose:
    //   Creates a new HDF5 File.
    // ---- Change History ----
    // 09/02/2011 -- Joe R. -- Initial implementation.
    void createHDF5File(const std::string& hdf5_filename);

    // ---------------------
    // createGroup
    // ---------------------
    // Purpose:
    //   Creates a group, overwriting anything at the
    //   specified object path if necessary.  Returns
    //   true on success.
    // ---- Change History ----
    // 07/15/2011 -- Joe R. -- Initial implementation.
    // 08/28/2011 -- Joe R. -- Throwing exception instead of using boolean ret val
    void createGroup(const std::string& hdf5_filename,
                     const std::string& hdf5_objname,
                     bool replace = false);

    // ---------------------
    // createDataSet
    // ---------------------
    // Purpose:
    //   Creates a dataset, overwriting anything at the
    //   specified object path if necessary.  Returns
    //   true on success.
    // ---- Change History ----
    // 07/15/2011 -- Joe R. -- Initial implementation.
    // 08/28/2011 -- Joe R. -- Throwing exception instead of using boolean ret val
    // 09/02/2011 -- Joe R. -- Adding replace arg
    void createDataSet(const std::string& hdf5_filename,
                       const std::string& hdf5_objname,
                       const BoundingBox& boundingBox,
                       const Dimension& dimension,
                       DataType dataType,
                       const bool replace = false,
                       const bool createGroups = true);

    // ---------------------
    // createDataSet
    // ---------------------
    // Purpose:
    //   Shortcut for the above without specifying
    //   boundingbox.
    // ---- Change History ----
    // 07/15/2011 -- Joe R. -- Initial implementation.
    // 08/28/2011 -- Joe R. -- Throwing exception instead of using boolean ret val
    inline void createDataSet(const std::string& hdf5_filename,
                              const std::string& hdf5_objname,
                              const Dimension& dimension,
                              DataType dataType,
                              const bool createGroups = true)
    {
      createDataSet(hdf5_filename,
                    hdf5_objname,
                    BoundingBox(0.0,0.0,0.0,
                                1.0,1.0,1.0),
                    dimension,
                    dataType,
                    createGroups);
    }

    // ---------------------
    // createDataSet
    // ---------------------
    // Purpose:
    //   Creates a string dataset and writes the specified
    //   string to it.
    // ---- Change History ----
    // 07/22/2011 -- Joe R. -- Initial implementation.
    // 08/28/2011 -- Joe R. -- Throwing exception instead of using boolean ret val
    void createDataSet(const std::string& hdf5_filename,
                       const std::string& hdf5_objname,
                       const std::string& value,
                       bool createGroups = true);

    // ---------------------
    // getObjectDimension
    // ---------------------
    // Purpose:
    //   Returns the dimensions of the specified object
    // ---- Change History ----
    // 07/17/2011 -- Joe R. -- Initial implementation.
    // 08/05/2011 -- Joe R. -- Renamed and generalized for both datasets and groups.
    Dimension getObjectDimension(const std::string& hdf5_filename,
                                 const std::string& hdf5_objname);

    // ---------------------
    // setObjectDimension
    // ---------------------
    // Purpose:
    //   Sets the dimensions of the specified object
    // ---- Change History ----
    // 08/26/2011 -- Joe R. -- Initial implementation.
    void setObjectDimension(const std::string& hdf5_filename,
                            const std::string& hdf5_objname,
                            const Dimension& dim);

    // ---------------------------------
    // getDataSetDimensionForBoundingBox
    // ---------------------------------
    // Purpose:
    //   Returns the dimensions of a sub-dataset defined by the
    //   bounding box.
    // ---- Change History ----
    // 09/04/2011 -- Joe R. -- Initial implementation.
    Dimension getDataSetDimensionForBoundingBox(const std::string& hdf5_filename,
                                                const std::string& hdf5_objname,
                                                const BoundingBox& subvolbox);    

    // ---------------------
    // getDataSetDimension
    // ---------------------
    // Purpose:
    //   Returns the dimensions of a sub-dataset defined by the
    //   bounding box.  The output dimension will be less than or
    //   equal to the maxdim.
    // ---- Change History ----
    // 07/22/2011 -- Joe R. -- Initial implementation.
    Dimension getDataSetDimension(const std::string& hdf5_filename,
                                  const std::string& hdf5_objname,
                                  const BoundingBox& subvolbox,
                                  const Dimension& maxdim = Dimension(256,256,256));

    // ---------------------
    // getObjectBoundingBox
    // ---------------------
    // Purpose:
    //   Returns the bounding box of the dataset
    // ---- Change History ----
    // 07/17/2011 -- Joe R. -- Initial implementation.
    // 08/05/2011 -- Joe R. -- Renamed and generalized for both datasets and groups.
    BoundingBox getObjectBoundingBox(const std::string& hdf5_filename,
                                     const std::string& hdf5_objname);

    // ---------------------
    // setObjectBoundingBox
    // ---------------------
    // Purpose:
    //   Sets the bounding box of the specified object
    // ---- Change History ----
    // 08/26/2011 -- Joe R. -- Initial implementation.
    void setObjectBoundingBox(const std::string& hdf5_filename,
                              const std::string& hdf5_objname,
                              const BoundingBox& boundingBox);

    // ---------------------
    // getDataSetMinimum
    // ---------------------
    // Purpose:
    //   Returns the minimum value of the dataset
    // ---- Change History ----
    // 07/17/2011 -- Joe R. -- Initial implementation.
    double getDataSetMinimum(const std::string& hdf5_filename,
                             const std::string& hdf5_objname);

    // ---------------------
    // getDataSetMaximum
    // ---------------------
    // Purpose:
    //   Returns the maximum value of the dataset
    // ---- Change History ----
    // 07/17/2011 -- Joe R. -- Initial implementation.
    double getDataSetMaximum(const std::string& hdf5_filename,
                             const std::string& hdf5_objname);


    // ---------------------
    // getDataSetInfo
    // ---------------------
    // Purpose:
    //   Returns the dataset info string
    // ---- Change History ----
    // 07/17/2011 -- Joe R. -- Initial implementation.
    std::string getDataSetInfo(const std::string& hdf5_filename,
                               const std::string& hdf5_objname);
    
    // ---------------------
    // getDataSetType
    // ---------------------
    // Purpose:
    //   Returns the dataset type
    // ---- Change History ----
    // 07/17/2011 -- Joe R. -- Initial implementation.
    DataType getDataSetType(const std::string& hdf5_filename,
                            const std::string& hdf5_objname);

    // ---------------
    // getChildObjects
    // ---------------
    // Purpose:
    //   Gets a list of child objects of the specified object. If no object
    //   specified, default is the root.
    // ---- Change History ----
    // 09/02/2011 -- Joe R. -- Initial implementation.
    // 09/17/2011 -- Joe R. -- Adding filter parameter.  A string isn't added
    //                         to the list if the filter IS NOT in the string
    std::vector<std::string> getChildObjects(const std::string& hdf5_filename,
                                             const std::string& hdf5_objname = "/",
                                             const std::string& filter = std::string());
    
    // ---------------------
    // readDataSet
    // ---------------------
    // Purpose:
    //   Read data into the memory location pointed to by values.
    // ---- Change History ----
    // 07/17/2011 -- Joe R. -- Initial implementation.
    // 08/26/2011 -- Joe R. -- Adding more detailed exception string
    template<class T>
    inline void readDataSet(const std::string& hdf5_filename,
                            const std::string& hdf5_objname,
                            uint64 off_x, uint64 off_y, uint64 off_z,
                            const Dimension& subvoldim,
                            T* values)
    {
      using namespace H5;

      cvcapp.log(10,boost::str(boost::format("%1%: %2%, %3%\n") 
                              % BOOST_CURRENT_FUNCTION
                              % hdf5_filename
                              % hdf5_objname));
      
      if(subvoldim.isNull())
        throw HDF5Exception("Null subvoldim");

      BoundingBox boundingBox = getObjectBoundingBox(hdf5_filename,
                                                     hdf5_objname);
      Dimension dimension = getObjectDimension(hdf5_filename,
                                               hdf5_objname);

      if(off_x + subvoldim[0] > dimension[0] ||
         off_y + subvoldim[1] > dimension[1] ||
         off_z + subvoldim[2] > dimension[2])
        throw InvalidHDF5File("Dimension and offset out of bounds");

      try
        {
          /*
           * Turn off the auto-printing when failure occurs so that we can
           * handle the errors appropriately
           */
          H5::Exception::dontPrint();

          ScopedLock lock(hdf5_filename,BOOST_CURRENT_FUNCTION);
          boost::shared_ptr<H5::H5File>  f;
          boost::shared_ptr<H5::DataSet> d;
        
          f = getH5File(hdf5_filename);
          d = getDataSet(*f,hdf5_objname,false);

	//calculate the subvolume boundingbox for the requested region
	BoundingBox subbbox
	  ( 
            boundingBox.minx + 
	    (boundingBox.maxx - boundingBox.minx)*
	    (off_x/(dimension.xdim-1)),
	    boundingBox.miny + 
	    (boundingBox.maxy - boundingBox.miny)*
	    (off_y/(dimension.ydim-1)),
	    boundingBox.minz + 
	    (boundingBox.maxz - boundingBox.minz)*
	    (off_z/(dimension.zdim-1)),
	    boundingBox.minx + 
	    (boundingBox.maxx - boundingBox.minx)*
	    ((off_x+subvoldim.xdim)/(dimension.xdim-1)),
	    boundingBox.miny + 
	    (boundingBox.maxy - boundingBox.miny)*
	    ((off_y+subvoldim.ydim)/(dimension.ydim-1)),
	    boundingBox.minz + 
	    (boundingBox.maxz - boundingBox.minz)*
	    ((off_z+subvoldim.zdim)/(dimension.zdim-1))
	  );

	  const int RANK = 3;
          hsize_t dimsf[RANK];     // dataset dimensions
          //NOTE: HDF5 dimensions are specified in opposite order (i.e. ZYX instead of XYZ)
          for(int i = 0; i < RANK; i++)
            dimsf[i] = subvoldim[RANK-1-i];
          DataSpace vol_dataspace( RANK, dimsf );
          vol_dataspace.selectAll(); //not sure if this is needed

          DataSpace filespace = d->getSpace();

          if(filespace.getSimpleExtentNdims() != RANK)
            throw InvalidHDF5File("invalid volume dataset rank!");

          hsize_t count[RANK];
          for(int i = 0; i < RANK; i++)
            count[i] = subvoldim[RANK-1-i];

          hsize_t offset[RANK] =
            { off_z, off_y, off_x };

          filespace.selectHyperslab(H5S_SELECT_SET, count, offset);
          d->read(values, 
                  getPredType(cvcapp.dataType<T>()),
                  vol_dataspace,
                  filespace);
        }
      catch( H5::Exception& error )
        {
          using namespace boost;
          throw HDF5Exception(str(format("filename: %s, object: %s, msg: %s")
                                  % hdf5_filename
                                  % hdf5_objname
                                  % error.getDetailMsg()));
        }
    } // readDataSet

    // ---------------------
    // readDataSet
    // ---------------------
    // Purpose:
    //   Read data into the memory location pointed to by values.  This version
    //   is for convenence when you have a pointer to a type that doesn't match the
    //   data you want to read, such as in the case of reading arbitrary data into
    //   memory via an unsigned char pointer (i.e. what VolMagick does)
    // ---- Change History ----
    // 08/05/2011 -- Joe R. -- Initial implementation.
    template<class T>
    inline void readDataSet(const std::string& hdf5_filename,
                            const std::string& hdf5_objname,
                            uint64 off_x, uint64 off_y, uint64 off_z,
                            const Dimension& subvoldim,
                            DataType dataType,
                            T* values)
    {
      cvcapp.log(10,boost::str(boost::format("%1%: %2%, %3%\n") 
                              % BOOST_CURRENT_FUNCTION
                              % hdf5_filename
                              % hdf5_objname));

      switch(dataType)
        {
        case UChar:
          {
            readDataSet(hdf5_filename, hdf5_objname,
                        off_x, off_y, off_z,
                        subvoldim,
                        reinterpret_cast<unsigned char*>(values));
          }
          break;
        case UShort:
          {
            readDataSet(hdf5_filename, hdf5_objname,
                        off_x, off_y, off_z,
                        subvoldim,
                        reinterpret_cast<unsigned short*>(values));
          }
          break;
        case UInt:
          {
            readDataSet(hdf5_filename, hdf5_objname,
                        off_x, off_y, off_z,
                        subvoldim,
                        reinterpret_cast<unsigned int*>(values));
          }
          break;
        case Float:
          {
            readDataSet(hdf5_filename, hdf5_objname,
                        off_x, off_y, off_z,
                        subvoldim,
                        reinterpret_cast<float*>(values));
          }
          break;
        case Double:
          {
            readDataSet(hdf5_filename, hdf5_objname,
                        off_x, off_y, off_z,
                        subvoldim,
                        reinterpret_cast<double*>(values));
          }
          break;
        case UInt64:
          {
            readDataSet(hdf5_filename, hdf5_objname,
                        off_x, off_y, off_z,
                        subvoldim,
                        reinterpret_cast<uint64*>(values));
          }
          break;
        case Char:
          {
            readDataSet(hdf5_filename, hdf5_objname,
                        off_x, off_y, off_z,
                        subvoldim,
                        reinterpret_cast<char*>(values));
          }
          break;
        }
    } // readDataSet
    
    // ---------------------
    // readDataSet
    // ---------------------
    // Purpose:
    //   Read data from specified dataset and returns a tuple containing
    //   a shared array of Ts and the actual dimensions of that array.
    //   actualDim will be less than or equal to maxdim.
    // ---- Change History ----
    // 07/22/2011 -- Joe R. -- Initial implementation.
    // 08/26/2011 -- Joe R. -- Adding more detailed exception string
    template<class T>
    inline boost::tuple< 
      boost::shared_array<T>,
      Dimension
    > readDataSet(const std::string& hdf5_filename,
                  const std::string& hdf5_objname,
                  const BoundingBox& subbox,
                  const Dimension& maxdim = Dimension(256,256,256))
    {
      using namespace H5;
      using namespace boost;

      cvcapp.log(10,str(format("%1%: %2%, %3%\n") 
                              % BOOST_CURRENT_FUNCTION
                              % hdf5_filename
                              % hdf5_objname));

      const int RANK = 3;
      BoundingBox boundingBox = getObjectBoundingBox(hdf5_filename,
                                                     hdf5_objname);
      Dimension dimension = getObjectDimension(hdf5_filename,
                                               hdf5_objname);
      double xspan = dimension.xdim == 0 ? 1.0 : (boundingBox.maxx-boundingBox.minx)/(dimension.xdim-1);
      double yspan = dimension.ydim == 0 ? 1.0 : (boundingBox.maxy-boundingBox.miny)/(dimension.ydim-1);
      double zspan = dimension.zdim == 0 ? 1.0 : (boundingBox.maxz-boundingBox.minz)/(dimension.zdim-1);

      Dimension actualDim;
      shared_array<T> data;  

      try
        {
          /*
           * Turn off the auto-printing when failure occurs so that we can
           * handle the errors appropriately
           */
          H5::Exception::dontPrint();

          ScopedLock lock(hdf5_filename,BOOST_CURRENT_FUNCTION);
          boost::shared_ptr<H5::H5File>  f;
          boost::shared_ptr<H5::DataSet> d;
        
          f = getH5File(hdf5_filename);
          d = getDataSet(*f,hdf5_objname,false);
      
          DataSpace filespace = d->getSpace();

          if(filespace.getSimpleExtentNdims() != RANK)
            throw InvalidHDF5File("invalid volume dataset rank!");

          Dimension fulldim(
            1+(subbox.maxx-subbox.minx)/xspan,
            1+(subbox.maxy-subbox.miny)/yspan,
            1+(subbox.maxz-subbox.minz)/zspan
          );

          if(fulldim.isNull())
            throw HDF5Exception("Null voxel selection");

          hsize_t off_x =
            (subbox.minx-boundingBox.minx)/xspan;
          hsize_t off_y =
            (subbox.miny-boundingBox.miny)/yspan;
          hsize_t off_z =
            (subbox.minz-boundingBox.minz)/zspan;
          
          hsize_t offset[RANK] =
            { off_z, off_y, off_x };
          
          for(int i = 0; i < RANK; i++)
            cvcapp.log(10,str(format("offset[%1%]: %2%\n") % i % offset[i]));
          
          hsize_t stride[RANK];
          for(int i = 0; i < RANK; i++)
            {
              stride[i] = fulldim[RANK-1-i]/maxdim[RANK-1-i];
              if(stride[i] == 0) stride[i]=1;
              if(fulldim[RANK-1-i] / stride[i] > maxdim[RANK-1-i])
                stride[i]++;
            }
	
          for(int i = 0; i < RANK; i++)
            cvcapp.log(10,str(format("stride[%1%]: %2%\n") % i % stride[i]));

          hsize_t count[RANK];
          for(int i = 0; i < RANK; i++)
            count[i] = fulldim[RANK-1-i]/stride[i];
          
          for(int i = 0; i < RANK; i++)
            cvcapp.log(10,str(format("count[%1%]: %2%\n") % i % count[i]));
          
          actualDim = Dimension(count[2],count[1],count[0]);
          data.reset(new T[actualDim.size()]);
          
          hsize_t dimsf[RANK];     // dataset dimensions
          for(int i = 0; i < RANK; i++)
            dimsf[i] = actualDim[RANK-1-i];
          DataSpace vol_dataspace( RANK, dimsf );
          vol_dataspace.selectAll();
          
          filespace.selectHyperslab(H5S_SELECT_SET, count, offset, stride);
          d->read(data.get(), 
                  getPredType(cvcapp.dataType<T>()),
                  vol_dataspace,
                  filespace);
        }
      catch(H5::Exception& error)
        {
          using namespace boost;
          throw HDF5Exception(str(format("filename: %s, object: %s, msg: %s")
                                  % hdf5_filename
                                  % hdf5_objname
                                  % error.getDetailMsg()));
        }

      return make_tuple(data,actualDim);
    }

    // ---------------------
    // readDataSet
    // ---------------------
    // Purpose:
    //   Read data from specified dataset and returns a tuple containing
    //   a shared array of uchars and the actual dimensions of that array.
    //   actualDim will be less than or equal to maxdim.  This version
    //   is for convenence when you need a pointer to a type that doesn't match the
    //   data you want to read, such as in the case of reading arbitrary data into
    //   memory via an unsigned char pointer (i.e. what VolMagick does)
    // ---- Change History ----
    // 08/26/2011 -- Joe R. -- Initial implementation.
    inline boost::tuple< 
      boost::shared_array<unsigned char>,
      Dimension
    > readDataSet(const std::string& hdf5_filename,
                  const std::string& hdf5_objname,
                  const BoundingBox& subbox,
                  DataType dataType,
                  const Dimension& maxdim = Dimension(256,256,256))
    {
      cvcapp.log(10,boost::str(boost::format("%1%: %2%, %3%\n") 
                              % BOOST_CURRENT_FUNCTION
                              % hdf5_filename
                              % hdf5_objname));

      Dimension dim;
      switch(dataType)
        {
        case UChar:
          {
            return readDataSet<unsigned char>(hdf5_filename, hdf5_objname,
                                              subbox, maxdim);
          }
          break;
        case UShort:
          {
            boost::shared_array<unsigned short> values;
            boost::tie(values,dim) =
              readDataSet<unsigned short>(hdf5_filename, hdf5_objname,
                                          subbox, maxdim);
            size_t len = dim.size()*DataTypeSizes[dataType];
            boost::shared_array<unsigned char> byte_values(new unsigned char[len]);
            memcpy(byte_values.get(),values.get(),len);
            return boost::make_tuple(byte_values,dim);
          }
          break;
        case UInt:
          {
            boost::shared_array<unsigned int> values;
            boost::tie(values,dim) =
              readDataSet<unsigned int>(hdf5_filename, hdf5_objname,
                                          subbox, maxdim);
            size_t len = dim.size()*DataTypeSizes[dataType];
            boost::shared_array<unsigned char> byte_values(new unsigned char[len]);
            memcpy(byte_values.get(),values.get(),len);
            return boost::make_tuple(byte_values,dim);
          }
          break;
        case Float:
          {
            boost::shared_array<float> values;
            boost::tie(values,dim) =
              readDataSet<float>(hdf5_filename, hdf5_objname,
                                 subbox, maxdim);
            size_t len = dim.size()*DataTypeSizes[dataType];
            boost::shared_array<unsigned char> byte_values(new unsigned char[len]);
            memcpy(byte_values.get(),values.get(),len);
            return boost::make_tuple(byte_values,dim);
          }
          break;
        case Double:
          {
            boost::shared_array<double> values;
            boost::tie(values,dim) =
              readDataSet<double>(hdf5_filename, hdf5_objname,
                                 subbox, maxdim);
            size_t len = dim.size()*DataTypeSizes[dataType];
            boost::shared_array<unsigned char> byte_values(new unsigned char[len]);
            memcpy(byte_values.get(),values.get(),len);
            return boost::make_tuple(byte_values,dim);
          }
          break;
        case UInt64:
          {
            boost::shared_array<uint64> values;
            boost::tie(values,dim) =
              readDataSet<uint64>(hdf5_filename, hdf5_objname,
                                 subbox, maxdim);
            size_t len = dim.size()*DataTypeSizes[dataType];
            boost::shared_array<unsigned char> byte_values(new unsigned char[len]);
            memcpy(byte_values.get(),values.get(),len);
            return boost::make_tuple(byte_values,dim);
          }
          break;
        case Char:
          {
            boost::shared_array<char> values;
            boost::tie(values,dim) =
              readDataSet<char>(hdf5_filename, hdf5_objname,
                                 subbox, maxdim);
            size_t len = dim.size()*DataTypeSizes[dataType];
            boost::shared_array<unsigned char> byte_values(new unsigned char[len]);
            memcpy(byte_values.get(),values.get(),len);
            return boost::make_tuple(byte_values,dim);
          }
          break;
        }
    }
    
    // ---------------------
    // readDataSet
    // ---------------------
    // Purpose:
    //   Read string dataset into string
    // ---- Change History ----
    // 07/22/2011 -- Joe R. -- Initial implementation.
    void readDataSet(const std::string& hdf5_filename,
                     const std::string& hdf5_objname,
                     std::string& value);

    // ---------------------
    // writeDataSet
    // ---------------------
    // Purpose:
    //   Writes data into the hdf5 file from memory location pointed to by values.
    // ---- Change History ----
    // 07/17/2011 -- Joe R. -- Initial implementation.
    // 08/26/2011 -- Joe R. -- Adding more detailed exception string
    template<class T>
    inline void writeDataSet(const std::string& hdf5_filename,
                             const std::string& hdf5_objname,
                             uint64 off_x, uint64 off_y, uint64 off_z,
                             const Dimension& values_dim,
                             const T* values,
                             T min_val, T max_val)
    {
      using namespace H5;

      cvcapp.log(10,boost::str(boost::format("%1%: %2%, %3%\n") 
                              % BOOST_CURRENT_FUNCTION
                              % hdf5_filename
                              % hdf5_objname));

      if(values_dim.isNull())
        throw HDF5Exception("Null subvoldim");

      BoundingBox boundingBox = getObjectBoundingBox(hdf5_filename,
                                                     hdf5_objname);
      Dimension dimension = getObjectDimension(hdf5_filename,
                                               hdf5_objname);
      
      try
        {
          ScopedLock lock(hdf5_filename,BOOST_CURRENT_FUNCTION);

          /*
           * Turn off the auto-printing when failure occurs so that we can
           * handle the errors appropriately
           */
          H5::Exception::dontPrint();

          boost::shared_ptr<H5::H5File>  f;
          boost::shared_ptr<H5::DataSet> d;
        
          f = getH5File(hdf5_filename);
          d = getDataSet(*f,hdf5_objname,false);

          const int RANK = 3;
          hsize_t dimsf[RANK];     // dataset dimensions
          for(int i = 0; i < RANK; i++)
            dimsf[i] = values_dim[RANK-1-i];
          DataSpace vol_dataspace( RANK, dimsf );
          vol_dataspace.selectAll();

          DataSpace filespace = d->getSpace();

          if(filespace.getSimpleExtentNdims() != RANK)
            throw InvalidHDF5File("invalid volume dataset rank!");

          hsize_t count[RANK];
          for(int i = 0; i < RANK; i++)
            count[i] = values_dim[RANK-1-i];
	
          hsize_t offset[RANK] =
            { off_z, off_y, off_x };

          filespace.selectHyperslab(H5S_SELECT_SET, count, offset);
          d->write(values, 
                   getPredType(cvcapp.dataType<T>()),
                   vol_dataspace,
                   filespace);

          //set the min/max if this volume changes it
          T file_min, file_max;
          getAttribute(*d,"min",file_min);
          getAttribute(*d,"max",file_max);
          if(file_min > min_val)
            setAttribute(*d,"min",min_val);
          if(file_max < max_val)
            setAttribute(*d,"max",max_val);
        }
      catch(H5::Exception& error)
        {
          using namespace boost;
          throw HDF5Exception(str(format("filename: %s, object: %s, msg: %s")
                                  % hdf5_filename
                                  % hdf5_objname
                                  % error.getDetailMsg()));
        }
    }

    // ---------------------
    // writeDataSet
    // ---------------------
    // Purpose:
    //   Writes data into the hdf5 file from memory location pointed to by values.
    //   Caculates and writes min/max values
    // ---- Change History ----
    // 07/17/2011 -- Joe R. -- Initial implementation.
    template<class T>
    inline void writeDataSet(const std::string& hdf5_filename,
                             const std::string& hdf5_objname,
                             uint64 off_x, uint64 off_y, uint64 off_z,
                             const Dimension& values_dim,
                             const T* values)
    {
      cvcapp.log(10,boost::str(boost::format("%1%: %2%, %3%\n") 
                              % BOOST_CURRENT_FUNCTION
                              % hdf5_filename
                              % hdf5_objname));

      uint64 values_size = values_dim.size();
      typedef std::list<int>::const_iterator iterator;
      std::pair< iterator, iterator > result = 
        boost::minmax_element(values, values+values_size);
      writeDataSet(hdf5_filename,
                   hdf5_objname,
                   off_x,off_y,off_z,
                   values_dim,
                   values,
                   *(result.first),
                   *(result.second));
    }

    // ---------------------
    // writeDataSet
    // ---------------------
    // Purpose:
    //   Writes data into the hdf5 file from memory location pointed to by values.
    //   Reinterprets the pointer to the specified data type.
    // ---- Change History ----
    // 08/28/2011 -- Joe R. -- Initial implementation.
    template<class T>
    inline void writeDataSet(const std::string& hdf5_filename,
                             const std::string& hdf5_objname,
                             uint64 off_x, uint64 off_y, uint64 off_z,
                             const Dimension& values_dim,
                             DataType dataType,
                             const T* values,
                             double min_val, double max_val)
    {
      cvcapp.log(10,boost::str(boost::format("%1%: %2%, %3%\n") 
                              % BOOST_CURRENT_FUNCTION
                              % hdf5_filename
                              % hdf5_objname));

      switch(dataType)
        {
        case UChar:
          {
            writeDataSet(hdf5_filename, hdf5_objname,
                         off_x, off_y, off_z,
                         values_dim, reinterpret_cast<const unsigned char*>(values),
                         (unsigned char)min_val,(unsigned char)max_val);
          }
          break;
        case UShort:
          {
            writeDataSet(hdf5_filename, hdf5_objname,
                         off_x, off_y, off_z,
                         values_dim, reinterpret_cast<const unsigned short*>(values),
                         (unsigned short)min_val,(unsigned short)max_val);
          }
          break;
        case UInt:
          {
            writeDataSet(hdf5_filename, hdf5_objname,
                         off_x, off_y, off_z,
                         values_dim, reinterpret_cast<const unsigned int*>(values),
                         (unsigned int)min_val,(unsigned int)max_val);
          }
          break;
        case Float:
          {
            writeDataSet(hdf5_filename, hdf5_objname,
                         off_x, off_y, off_z,
                         values_dim, reinterpret_cast<const float*>(values),
                         float(min_val),float(max_val));
          }
          break;
        case Double:
          {
            writeDataSet(hdf5_filename, hdf5_objname,
                         off_x, off_y, off_z,
                         values_dim, reinterpret_cast<const double*>(values),
                         min_val,max_val);
          }
          break;
        case UInt64:
          {
            writeDataSet(hdf5_filename, hdf5_objname,
                         off_x, off_y, off_z,
                         values_dim, reinterpret_cast<const uint64*>(values),
                         uint64(min_val),uint64(max_val));
          }
          break;
        case Char:
          {
            writeDataSet(hdf5_filename, hdf5_objname,
                         off_x, off_y, off_z,
                         values_dim, reinterpret_cast<const char*>(values),
                         char(min_val),char(max_val));
          }
          break;
        }
    }  //writeDataSet

    // ---------------------
    // writeDataSet
    // ---------------------
    // Purpose:
    //   Read string dataset into string
    // ---- Change History ----
    // 07/22/2011 -- Joe R. -- Initial implementation.
    void writeDataSet(const std::string& hdf5_filename,
                      const std::string& hdf5_objname,
                      const std::string& value);

    // ---------------------
    // getAttribute
    // ---------------------
    // Purpose:
    //   Reads a 1D array of T elements from an
    //   attribute of an object inside an hdf file.
    // ---- Change History ----
    // 07/15/2011 -- Joe R. -- Initial implementation.
    // 08/26/2011 -- Joe R. -- Adding more detailed exception string
    template<class T>
    inline void getAttribute(const std::string& hdf5_filename,
                             const std::string& hdf5_objname,
                             const std::string& attribname,
                             size_t len,
                             T* values)
    {
      cvcapp.log(10,boost::str(boost::format("%1%: %2%, %3%, %4%\n") 
                              % BOOST_CURRENT_FUNCTION
                              % hdf5_filename
                              % hdf5_objname
                              % attribname));

      bool isgroup = isGroup(hdf5_filename,hdf5_objname);
      bool isdataset = isDataSet(hdf5_filename,hdf5_objname);

      try
        {
          /*
           * Turn off the auto-printing when failure occurs so that we can
           * handle the errors appropriately
           */
          H5::Exception::dontPrint();


          ScopedLock lock(hdf5_filename,BOOST_CURRENT_FUNCTION);
          boost::shared_ptr<H5::H5File>  f;
          boost::shared_ptr<H5::Group>   g;
          boost::shared_ptr<H5::DataSet> d;
        
          H5::H5Object *obj = NULL;
          if(isgroup)
            {
              f = getH5File(hdf5_filename);
              g = getGroup(*f,hdf5_objname,false);
              obj = g.get();
            }
          else if(isdataset)
            {
              f = getH5File(hdf5_filename);
              d = getDataSet(*f,hdf5_objname,false);
              obj = d.get();
            }

          if(!obj)
            throw HDF5Exception("Unknown object type!");

          getAttribute(*obj,attribname,len,values);
        }
      catch( H5::Exception& error )
        {
          using namespace boost;
          throw HDF5Exception(str(format("filename: %s, object: %s, attrib: %s, msg: %s")
                                  % hdf5_filename
                                  % hdf5_objname
                                  % attribname
                                  % error.getDetailMsg()));
        }
    }

    // ---------------------
    // getAttribute
    // ---------------------
    // Purpose:
    //  Forcing DataType enum to use uint64
    // ---- Change History ----
    // 08/26/2011 -- Joe R. -- Initial implementation.
    template<>
    inline void getAttribute<DataType>(const std::string& hdf5_filename,
                                       const std::string& hdf5_objname,
                                       const std::string& attribname,
                                       size_t len,
                                       DataType* values)
    {
      cvcapp.log(10,boost::str(boost::format("%1%: %2%, %3%, %4%\n") 
                              % BOOST_CURRENT_FUNCTION
                              % hdf5_filename
                              % hdf5_objname
                              % attribname));

      std::vector<uint64> dt_values(len);
      getAttribute(hdf5_filename, hdf5_objname, attribname, len, &(dt_values[0]));
      for(size_t i = 0; i < len; i++)
        values[i] = DataType(dt_values[i]);
    }                           

    // ---------------------
    // getAttribute
    // ---------------------
    // Purpose:
    //   Reads an attribute of type T with specified 
    //   name from the specified object.
    // ---- Change History ----
    // 07/15/2011 -- Joe R. -- Initial implementation.
    template<class T>
    inline void getAttribute(const std::string& hdf5_filename,
                             const std::string& hdf5_objname,
                             const std::string& attribname,
                             T& value)
    {
      cvcapp.log(10,boost::str(boost::format("%1%: %2%, %3%, %4%\n") 
                              % BOOST_CURRENT_FUNCTION
                              % hdf5_filename
                              % hdf5_objname
                              % attribname));

      getAttribute(hdf5_filename,
                   hdf5_objname,
                   attribname,
                   1, 
                   &value);
    }

    // ---------------------
    // getAttribute
    // ---------------------
    // Purpose:
    //   Gets a string attribute from an object.
    // ---- Change History ----
    // 07/15/2011 -- Joe R. -- Initial implementation.
    // 08/26/2011 -- Joe R. -- Adding more detailed exception string
    template<>
    inline void getAttribute<std::string>(const std::string& hdf5_filename,
                                          const std::string& hdf5_objname,
                                          const std::string& attribname,
                                          std::string& value)
    {
      cvcapp.log(10,boost::str(boost::format("%1%: %2%, %3%, %4%\n") 
                              % BOOST_CURRENT_FUNCTION
                              % hdf5_filename
                              % hdf5_objname
                              % attribname));

      bool isgroup = isGroup(hdf5_filename,hdf5_objname);
      bool isdataset = isDataSet(hdf5_filename,hdf5_objname);

      try
        {
          /*
           * Turn off the auto-printing when failure occurs so that we can
           * handle the errors appropriately
           */
          H5::Exception::dontPrint();

          ScopedLock lock(hdf5_filename,BOOST_CURRENT_FUNCTION);
          boost::shared_ptr<H5::H5File>  f;
          boost::shared_ptr<H5::Group>   g;
          boost::shared_ptr<H5::DataSet> d;
        
          H5::H5Object *obj = NULL;
          if(isgroup)
            {
              f = getH5File(hdf5_filename);
              g = getGroup(*f,hdf5_objname,false);
              obj = g.get();
            }
          else if(isdataset)
            {
              f = getH5File(hdf5_filename);
              d = getDataSet(*f,hdf5_objname,false);
              obj = d.get();
            }

          if(!obj)
            throw HDF5Exception("Unknown object type!");

          getAttribute(*obj,attribname,value);
        }
      catch( H5::Exception& error )
        {
          using namespace boost;
          throw HDF5Exception(str(format("filename: %s, object: %s, attrib: %s, msg: %s")
                                  % hdf5_filename
                                  % hdf5_objname
                                  % attribname
                                  % error.getDetailMsg()));
        }
    }

    // ---------------------
    // setAttribute
    // ---------------------
    // Purpose:
    //   Sets a 1D array of T elements to an
    //   attribute of an object inside an hdf file.
    // ---- Change History ----
    // 07/15/2011 -- Joe R. -- Initial implementation.
    // 08/26/2011 -- Joe R. -- Adding more detailed exception string
    template<class T>
    inline void setAttribute(const std::string& hdf5_filename,
                             const std::string& hdf5_objname,
                             const std::string& attribname,
                             size_t len,
                             const T* values)
    {
      cvcapp.log(10,boost::str(boost::format("%1%: %2%, %3%, %4%\n") 
                              % BOOST_CURRENT_FUNCTION
                              % hdf5_filename
                              % hdf5_objname
                              % attribname));

      bool isgroup = isGroup(hdf5_filename,hdf5_objname);
      bool isdataset = isDataSet(hdf5_filename,hdf5_objname);

      try
        {
          /*
           * Turn off the auto-printing when failure occurs so that we can
           * handle the errors appropriately
           */
          H5::Exception::dontPrint();

          ScopedLock lock(hdf5_filename,BOOST_CURRENT_FUNCTION);
          boost::shared_ptr<H5::H5File>  f;
          boost::shared_ptr<H5::Group>   g;
          boost::shared_ptr<H5::DataSet> d;
        
          H5::H5Object *obj = NULL;
          if(isgroup)
            {
              f = getH5File(hdf5_filename);
              g = getGroup(*f,hdf5_objname,false);
              obj = g.get();
            }
          else if(isdataset)
            {
              f = getH5File(hdf5_filename);
              d = getDataSet(*f,hdf5_objname,false);
              obj = d.get();
            }

          if(!obj)
            throw HDF5Exception("Unknown object type!");

          setAttribute(*obj,attribname,len,values);
        }
      catch( H5::Exception& error )
        {
          using namespace boost;
          throw HDF5Exception(str(format("filename: %s, object: %s, attrib: %s, msg: %s")
                                  % hdf5_filename
                                  % hdf5_objname
                                  % attribname
                                  % error.getDetailMsg()));
        }
    }

    // ---------------------
    // setAttribute
    // ---------------------
    // Purpose:
    //   Forcing DataType enum to use uint64 type
    // ---- Change History ----
    // 08/26/2011 -- Joe R. -- Initial implementation.
    template<>
    inline void setAttribute<DataType>(const std::string& hdf5_filename,
                                       const std::string& hdf5_objname,
                                       const std::string& attribname,
                                       size_t len,
                                       const DataType* values)
    {
      cvcapp.log(10,boost::str(boost::format("%1%: %2%, %3%, %4%\n") 
                              % BOOST_CURRENT_FUNCTION
                              % hdf5_filename
                              % hdf5_objname
                              % attribname));

      std::vector<uint64> dt_values(len);
      for(size_t i = 0; i < len; i++)
        dt_values[i] = uint64(values[i]);
      setAttribute(hdf5_filename, hdf5_objname, attribname, len, &(dt_values[0]));
    }

    // ---------------------
    // setAttribute
    // ---------------------
    // Purpose:
    //   Sets an attribute of type T with specified 
    //   name to the specified object.
    // ---- Change History ----
    // 07/15/2011 -- Joe R. -- Initial implementation.
    template<class T>
    inline void setAttribute(const std::string& hdf5_filename,
                             const std::string& hdf5_objname,
                             const std::string& attribname,
                             const T& value)
    {
      cvcapp.log(10,boost::str(boost::format("%1%: %2%, %3%, %4%\n") 
                              % BOOST_CURRENT_FUNCTION
                              % hdf5_filename
                              % hdf5_objname
                              % attribname));

      setAttribute(hdf5_filename,
                   hdf5_objname,
                   attribname,
                   1, 
                   &value);
    }

    // ---------------------
    // setAttribute
    // ---------------------
    // Purpose:
    //   Sets a string attribute to an object.
    // ---- Change History ----
    // 07/15/2011 -- Joe R. -- Initial implementation.
    // 08/26/2011 -- Joe R. -- Adding more detailed exception string
    template<>
    inline void setAttribute<std::string>(const std::string& hdf5_filename,
                                          const std::string& hdf5_objname,
                                          const std::string& attribname,
                                          const std::string& value)
    {
      cvcapp.log(10,boost::str(boost::format("%1%: %2%, %3%, %4%\n") 
                              % BOOST_CURRENT_FUNCTION
                              % hdf5_filename
                              % hdf5_objname
                              % attribname));

      bool isgroup = isGroup(hdf5_filename,hdf5_objname);
      bool isdataset = isDataSet(hdf5_filename,hdf5_objname);

      try
        {
          /*
           * Turn off the auto-printing when failure occurs so that we can
           * handle the errors appropriately
           */
          H5::Exception::dontPrint();

          ScopedLock lock(hdf5_filename,BOOST_CURRENT_FUNCTION);
          boost::shared_ptr<H5::H5File>  f;
          boost::shared_ptr<H5::Group>   g;
          boost::shared_ptr<H5::DataSet> d;
        
          H5::H5Object *obj = NULL;
          if(isgroup)
            {
              f = getH5File(hdf5_filename);
              g = getGroup(*f,hdf5_objname,false);
              obj = g.get();
            }
          else if(isdataset)
            {
              f = getH5File(hdf5_filename);
              d = getDataSet(*f,hdf5_objname,false);
              obj = d.get();
            }

          if(!obj)
            throw HDF5Exception("Unknown object type!");

          setAttribute(*obj,attribname,value);
        }
      catch( H5::Exception& error )
        {
          using namespace boost;
          throw HDF5Exception(str(format("filename: %s, object: %s, attrib: %s, msg: %s")
                                  % hdf5_filename
                                  % hdf5_objname
                                  % attribname
                                  % error.getDetailMsg()));
        }
    }

    // ---------------------
    // setAttribute
    // ---------------------
    // Purpose:
    //   Sets a string attribute to an object via a char*
    // ---- Change History ----
    // 08/28/2011 -- Joe R. -- Initial implementation.
    inline void setAttribute(const std::string& hdf5_filename,
                             const std::string& hdf5_objname,
                             const std::string& attribname,
                             const char* value)
    {
      cvcapp.log(10,boost::str(boost::format("%1%: %2%, %3%, %4%\n") 
                              % BOOST_CURRENT_FUNCTION
                              % hdf5_filename
                              % hdf5_objname
                              % attribname));

      setAttribute(hdf5_filename, hdf5_objname,
                   attribname, std::string(value));
    }
  }
}

#endif

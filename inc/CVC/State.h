/*
  Copyright 2012 The University of Texas at Austin

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

/* $Id: State.h 5559 2012-05-11 21:43:22Z transfix $ */

#ifndef __CVC_STATE_H__
#define __CVC_STATE_H__

#include <CVC/Namespace.h>
#include <CVC/Types.h>
#include <CVC/App.h>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/function.hpp>

namespace CVC_NAMESPACE
{
  // ----------
  // CVC::State
  // ----------
  // Purpose: 
  //   Central program state manangement.  Provides a tree
  //   to which property values and arbitrary data can be attached.
  //   Written to be thread safe and to be used also as a thread
  //   messaging system.  With XmlRPC that messaging can extend to
  //   threads in other processes and nodes on the network.
  // ---- Change History ----
  // 02/18/2012 -- Joe R. -- Initial implementation.
  // 03/02/2012 -- Joe R. -- Added touch()
  // 03/15/2012 -- Joe R. -- Added initialized flag.
  // 03/16/2012 -- Joe R. -- Added reset(), ptree() and traverse()
  // 03/30/2012 -- Joe R. -- Added comment and hidden field.
  // 03/31/2012 -- Joe R. -- Added dataTypeName().
  class State
  {
  public:  
    typedef boost::shared_ptr<State> StatePtr;
    typedef std::map<std::string,StatePtr> ChildMap;
    typedef boost::function<void (std::string)> TraversalUnaryFunc;

    static const std::string SEPARATOR;

    virtual ~State();

    // ***** Main API

    //Use instance() to grab a reference to the singleton application object.
    static State& instance();

    const std::string& name()   const { return _name;   }
    const State*       parent() const { return _parent; }

    //return's the parent's fullName
    std::string parentName() const { 
      std::string tmp;
      return parent() ? 
        (!parent()->name().empty() ?
         ((tmp=parent()->parentName()).empty()?
          parent()->name() : 
          tmp + SEPARATOR + parent()->name()) 
         : "")
        : "";
    }

    std::string fullName() const { 
      std::string pn = parentName();
      return pn.empty() ?
        name() :
        pn + SEPARATOR + name();
    }

    boost::posix_time::ptime lastMod();

    std::string value();
    std::string valueTypeName();
    std::vector<std::string> values(bool unique = false); //shortcut for comma separated values in value()
    State& value(const std::string& v, bool setValueType = true);
    template <class T> T value() { return boost::lexical_cast<T>(value()); }
    template <class T> State& value(const T& v) {
      {
        boost::mutex::scoped_lock lock(_mutex);
        _valueTypeName = cvcapp.dataTypeName<T>();
      }
      return value(boost::lexical_cast<std::string>(v),false);
    }
    Signal valueChanged;

    boost::any data();
    State& data(const boost::any&);

    template<class T>
    T data()
    {
      return boost::any_cast<T>(data());
    }

    template<class T>
    bool isData()
    {
      try
	{
	  T val = data<T>();
	}
      catch(std::exception& e)
	{
	  return false;
	}
      return true;
    }

    std::string dataTypeName();

    Signal dataChanged;

    State& operator()(const std::string& childname = std::string());
    std::vector<std::string> children(const std::string& re = std::string());
    size_t numChildren();
    MapChangeSignal childChanged;

    operator std::string(){ return value(); }

    Signal destroyed;

    void touch();

    bool initialized() const { return _initialized; }

    //like propertyData from CVC::App
    template<class T>
    std::vector<T> valueData(bool uniqueElements = false)
    {
      using namespace std;
      using namespace boost;
      using namespace boost::algorithm;
      vector<string> vals = values(uniqueElements);
      vector<T> ret_data;
      BOOST_FOREACH(string dkey, vals)
        {
          trim(dkey);
          if(CVC_NAMESPACE::State::instance()(dkey).isData<T>())
            ret_data.push_back(CVC_NAMESPACE::State::instance()(dkey).data<T>());
        }
      return ret_data;
    }

    void reset();

    //converting to and from a boost property tree.  Useful for saving and restoring state.
    boost::property_tree::ptree ptree();
    operator boost::property_tree::ptree(){ return ptree(); }
    void ptree(const boost::property_tree::ptree&);

    void save(const std::string& filename);
    void restore(const std::string& filename);

    void traverse(TraversalUnaryFunc func, const std::string& re = std::string());
    Signal traverseEnter;
    Signal traverseExit;

    std::string comment();
    State& comment(const std::string& c);
    Signal commentChanged;

    bool hidden();
    State& hidden(bool h);
    Signal hiddenChanged;

  protected:
    State(const std::string& n = std::string(),
          const State* p = NULL);

    void notifyParent(const std::string& childname);
    void notifyXmlRpc();

    boost::mutex                     _mutex;
    boost::posix_time::ptime         _lastMod;

    std::string                      _name;
    const State*                     _parent;

    std::string                      _value;
    std::string                      _valueTypeName;
    boost::any                       _data;
    std::string                      _comment;
    bool                             _hidden;
    ChildMap                         _children;

    bool                             _initialized;
    
    static StatePtr instancePtr();
    static StatePtr                  _instance;
    static boost::mutex              _instanceMutex;
  private:
    State(const State&);
  };
}

//Shorthand to access the App object from anywhere
#define cvcstate CVC_NAMESPACE::State::instance()

#endif

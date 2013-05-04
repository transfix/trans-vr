/*
  Copyright 2005-2012 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolUtils.

  VolUtils is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolUtils is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: cvcstate_test.cpp 5355 2012-04-06 22:16:56Z transfix $ */

#include <CVC/App.h>
#include <CVC/State.h>

#include <VolMagick/VolMagick.h>

#include <boost/asio/ip/host_name.hpp>

int main(int argc, char **argv)
{
  cvcapp.log(1,"start\n");

  if(argc > 2)
    {
      cvcstate.restore(argv[1]);
      //TODO: print what was restored and return.
    }
  
  cvcstate("__system.xmlrpc.hosts").value("neutron.ices.utexas.edu");
  //cvcstate("__system.xmlrpc.hosts").value("localhost");

  cvcstate("cvc.volumes")("volume0").value("stuff");
  cvcstate("cvc.volumes").value("whatever");
  cvcstate("cvc.volumes")("volume0").data(VolMagick::Volume());

#if 1
  cvcapp.log(1,"cvc.volumes.volume0 = " +
             cvcstate("cvc.volumes.volume0").value() + "\n");
  cvcapp.log(1,"cvc.volumes = " +
             cvcstate("cvc.volumes").value() + "\n");
  cvcapp.log(1,"cvc.volumes.volume0 parentName = " 
             + cvcstate("cvc.volumes.volume0").parentName() + "\n");
  cvcapp.log(1,"cvc.volumes parentName = " 
             + cvcstate("cvc.volumes").parentName() + "\n");
  cvcapp.log(1,"cvc parentName = " 
             + cvcstate("cvc").parentName() + "\n");
  cvcapp.log(1,"root parentName = " 
             + cvcstate().parentName() + "\n");
  cvcapp.log(1,"cvc.volumes.volume1.test fullName = " 
             + cvcstate("cvc.volumes.volume1.test").fullName() + "\n");
  cvcapp.log(1,"hostname = " + boost::asio::ip::host_name() + "\n");
#endif

#if 1
  //sleep(5);

  cvcstate("shared.whatever").value("stuff!");

  //sleep(5);
#endif

#if 1
  std::vector<std::string> children = cvcstate().children();
  BOOST_FOREACH(std::string child, children)
    cvcapp.log(1,child+" = "+cvcstate(child).value()+"\n");
#endif

  cvcstate.save("test.info");

  cvcapp.wait();
  return 0;
}

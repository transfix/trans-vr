#ifndef CGAL_BSO_RATIONAL_NT_H
#define CGAL_BSO_RATIONAL_NT_H

#include <CGAL/basic.h>

#ifdef CGAL_USE_GMP

  // GMP is installed. Use the GMP rational number-type.
  #include <CGAL/Gmpq.h>

  typedef CGAL::Gmpq                                    Bso_number_type;

#else

  // GMP is not installed. Use CGAL's exact rational number-type.
  #include <CGAL/MP_Float.h>
  #include <CGAL/Quotient.h>

  typedef CGAL::Quotient<CGAL::MP_Float>                Bso_number_type;

#endif

#endif

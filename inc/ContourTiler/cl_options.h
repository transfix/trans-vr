#ifndef __CL_TILER_OPTIONS_H__
#define __CL_TILER_OPTIONS_H__

#include <ContourTiler/Tiler_options.h>
#include <ContourTiler/common.h>
#include <boost/unordered_set.hpp>
#include <vector>

CONTOURTILER_BEGIN_NAMESPACE

struct cl_options {
  cl_options()
      : ser(true), ser2(false), start(1), end(2), smoothing_factor(-1),
        print_components(false), td_int_remove_only(false) {}

  std::vector<std::string> files;
  bool ser, ser2;
  Tiler_options options;
  boost::unordered_set<std::string> components;
  boost::unordered_set<std::string> components_skip;
  int start, end;
  int smoothing_factor;
  bool print_components;
  bool td_int_remove_only;
};

cl_options cl_parse(int argc, char **argv);
cl_options cl_parse(const std::vector<std::string> &argv);

CONTOURTILER_END_NAMESPACE

#endif

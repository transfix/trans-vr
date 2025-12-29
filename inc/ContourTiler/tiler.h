#ifndef __TILER_H__
#define __TILER_H__

#include <ContourTiler/Color.h>
#include <ContourTiler/Contour.h>
#include <ContourTiler/Slice.h>
#include <ContourTiler/Tile.h>
#include <ContourTiler/Tiler_options.h>
#include <ContourTiler/Tiler_workspace.h>
#include <boost/shared_ptr.hpp>

CONTOURTILER_BEGIN_NAMESPACE

template <typename Slice_iter>
void tile(Slice_iter slices_begin, Slice_iter slices_end,
          const boost::unordered_map<string, Color> &comp2color,
          const Tiler_options &options);

CONTOURTILER_END_NAMESPACE

#endif

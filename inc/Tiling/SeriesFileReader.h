#ifndef __SERIESFILEREADER_H__
#define __SERIESFILEREADER_H__

#include <list>
#include <string>
#include <VolumeGridRover/SurfRecon.h>

#include <VolMagick/VolMagick.h>

namespace SeriesFileReader
{
  extern std::list<SurfRecon::ContourPtr> readSeries(const std::string& filename, const VolMagick::VolumeFileInfo& volinfo);
  //double thickness = 1.0, double scale = 1.0);
};

#endif

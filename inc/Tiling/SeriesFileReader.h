#ifndef __SERIESFILEREADER_H__
#define __SERIESFILEREADER_H__

#include <VolMagick/VolMagick.h>
#include <VolumeGridRover/SurfRecon.h>
#include <list>
#include <string>

namespace SeriesFileReader {
extern std::list<SurfRecon::ContourPtr>
readSeries(const std::string &filename,
           const VolMagick::VolumeFileInfo &volinfo);
// double thickness = 1.0, double scale = 1.0);
}; // namespace SeriesFileReader

#endif

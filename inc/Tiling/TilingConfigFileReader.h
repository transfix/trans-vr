#ifndef __TILINGCONFIGFILEREADER_H__
#define __TILINGCONFIGFILEREADER_H__

#include <VolMagick/VolMagick.h>
#include <VolumeGridRover/SurfRecon.h>
#include <list>
#include <string>

namespace TilingConfigFileReader {
extern std::list<SurfRecon::ContourPtr>
readConfig(const std::string &filename,
           const VolMagick::VolumeFileInfo &volinfo);
};

#endif

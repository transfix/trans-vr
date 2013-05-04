#ifndef __TILINGCONFIGFILEREADER_H__
#define __TILINGCONFIGFILEREADER_H__

#include <list>
#include <string>
#include <VolumeGridRover/SurfRecon.h>

#include <VolMagick/VolMagick.h>

namespace TilingConfigFileReader
{
  extern std::list<SurfRecon::ContourPtr> readConfig(const std::string& filename, const VolMagick::VolumeFileInfo& volinfo);
};

#endif

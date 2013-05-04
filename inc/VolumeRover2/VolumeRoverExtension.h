#ifndef __VOLUMEROVEREXTENSION_H__
#define __VOLUMEROVEREXTENSION_H__

#ifdef USING_VOLUMEROVER_NAMESPACE
#ifndef VOLUMEROVER_NAMESPACE
#define VOLUMEROVER_NAMESPACE VolumeRover
#endif

namespace VOLUMEROVER_NAMESPACE
{
#endif

  // --------------------
  // VolumeRoverExtension
  // --------------------
  // Purpose:
  //   Extensions (libraries/plugins) are initialized through
  //   this interface.
  class VolumeRoverExtension
  {
  public:
    VolumeRoverExtension() {}
    virtual ~VolumeRoverExtension() {}
    virtual void init() = 0;
  };
  
#ifdef USING_VOLUMEROVER_NAMESPACE
}
#endif

#endif

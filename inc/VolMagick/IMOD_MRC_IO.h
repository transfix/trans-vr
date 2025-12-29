#ifndef __IMOD_MRC_H__
#define __IMOD_MRC_H__

#include <VolMagick/MRC_IO.h>

namespace VolMagick {
// -----------
// IMOD_MRC_IO
// -----------
// Purpose:
//   Uses IMOD's reading routines to read troublesome MRC files that the
//   regular MRC_IO doesn't seem to support.
// ---- Change History ----
// 11/20/2009 -- Joe R. -- Initial implementation
struct IMOD_MRC_IO : public MRC_IO {
  // ------------------------
  // IMOD_MRC_IO::IMOD_MRC_IO
  // ------------------------
  // Purpose:
  //   Initializes the extension list and id.
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation.
  IMOD_MRC_IO();

  // ------------------------------
  // IMOD_MRC_IO::getVolumeFileInfo
  // ------------------------------
  // Purpose:
  //   Writes to a structure containing all info that VolMagick needs
  //   from a volume file.
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation.
  virtual void getVolumeFileInfo(VolumeFileInfo::Data &data,
                                 const std::string &filename) const;

  // ---------------------------
  // IMOD_MRC_IO::readVolumeFile
  // ---------------------------
  // Purpose:
  //   Writes to a Volume object after reading from a volume file.
  // ---- Change History ----
  // 11/20/2009 -- Joe R. -- Initial implementation.
  virtual void readVolumeFile(Volume &vol, const std::string &filename,
                              unsigned int var, unsigned int time,
                              uint64 off_x, uint64 off_y, uint64 off_z,
                              const Dimension &subvoldim) const;
};
} // namespace VolMagick

#endif

#ifndef FILTER_H
#define FILTER_H

namespace Tiling {

void median_filter33(float *buf, int dimx, int dimy);

void lowpass_filter33(float *buf, int dimx, int dimy);

}; // namespace Tiling

#endif

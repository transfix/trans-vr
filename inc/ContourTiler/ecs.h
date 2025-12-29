#ifndef __ECS_H__
#define __ECS_H__

#include <ContourTiler/common.h>
#include <limits>
#include <vector>

CONTOURTILER_BEGIN_NAMESPACE

struct ecs_Bbox_3 {
  ecs_Bbox_3() {
    double imn = std::numeric_limits<double>::max();
    double imx = -std::numeric_limits<double>::max();
    init(imn, imn, imn, imx, imx, imx);
    for (int i = 0; i < 6; ++i)
      apply[i] = true;
  }
  ecs_Bbox_3(double xmin, double ymin, double zmin, double xmax, double ymax,
             double zmax) {
    init(xmin, ymin, zmin, xmax, ymax, zmax);
    for (int i = 0; i < 6; ++i)
      apply[i] = true;
  }
  void init(double xmin, double ymin, double zmin, double xmax, double ymax,
            double zmax) {
    mins[0] = xmin;
    mins[1] = ymin;
    mins[2] = zmin;
    maxs[0] = xmax;
    maxs[1] = ymax;
    maxs[2] = zmax;
  }
  void add(const Point_3 &p);
  void contract() {
    for (int i = 0; i < 3; ++i) {
      double w = maxs[i] - mins[i];
      mins[i] += w * .01;
      maxs[i] -= w * .01;
    }
  }
  double min(int idx) const { return mins[idx]; }
  double max(int idx) const { return maxs[idx]; }
  double val(int idx) const {
    if (idx < 3)
      return min(idx);
    return max(idx - 3);
  }

  double mins[3], maxs[3];
  bool apply[6];
};

void process_ecs(const std::vector<std::string> &filenames, std::string outfn,
                 ecs_Bbox_3 bb, bool bb_init, bool crop_only);

CONTOURTILER_END_NAMESPACE

#endif

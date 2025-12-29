/* $Id: volcombine.cpp 4134 2011-05-19 15:25:18Z arand $ */

#include <VolMagick/VolMagick.h>
#include <VolMagick/VolumeCache.h>
#include <VolMagick/endians.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char **argv) {

  if (argc != 6 && argc != 4) {
    cerr
        << "Usage: " << argv[0]
        << " <input volume file1> <input volume file2> [background threshold "
           "for vol1] [background threshold for vol2] <output txtfile>"
        << endl;
    return 1;
  }

  try {
    VolMagick::Volume invol1, invol2;

    VolMagick::readVolumeFile(invol1, argv[1]); // read the first volume file
    VolMagick::readVolumeFile(invol2, argv[2]); // read the second volume file

    if (invol1.dimension() != invol2.dimension()) {
      cout << "These two volumes' dimension are not the same. Correlation "
              "cannot be computed."
           << endl;
      return 1;
    }

    float background1, background2;
    background1 = invol1.min();
    background2 = invol2.min();

    if (argc == 6) {
      background1 = atof(argv[3]);
      background2 = atof(argv[4]);
    }

    long double corr = 0.0;
    for (int k = 0; k < invol1.ZDim(); k++)
      for (int j = 0; j < invol1.YDim(); j++)
        for (int i = 0; i < invol1.XDim(); i++) {
          if (invol1(i, j, k) >= background1 &&
              invol2(i, j, k) >= background2)
            corr += invol1(i, j, k) * invol2(i, j, k);
        }

    FILE *fp = fopen(argv[argc - 1], "w");
    if (fp == NULL) {
      cout << "File cannot be written" << endl;
      return 1;
    }
    fprintf(fp, "%Lf\n", corr);

    fclose(fp);

  } catch (VolMagick::Exception &e) {
    cerr << e.what() << endl;
  } catch (std::exception &e) {
    cerr << e.what() << endl;
  }

  return 0;
}

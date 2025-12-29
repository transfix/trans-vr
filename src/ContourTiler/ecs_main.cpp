// This code clips polyhedra with an axis-aligned bounding box and then
// finds the complement polyhedron.
//
// The bounding box must be axis aligned becaues of numerical issues.

#include <ContourTiler/common.h>
#include <ContourTiler/ecs.h>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <log4cplus/configurator.h>
#include <log4cplus/fileappender.h>
#include <log4cplus/logger.h>
#include <log4cplus/loglevel.h>

using namespace std;
using namespace CONTOURTILER_NAMESPACE;

void print_usage() {
  // cout << "Usage: ecs [-p x] [-b] [-a] [-o outfile] xmin ymin zmin xmax
  // ymax zmax [infiles] outfile" << endl;
  cout << "Usage: ecs [-b xmin ymin zmin xmax ymax zmax] [-c] [infiles] "
          "outfile"
       << endl;
  cout << "Create an extracellular space model from input files." << endl;
  cout << endl;
  cout << "  xmin...       bounding box to clip meshes" << endl;
  cout << "  -c            don\'t find ECS, just crop" << endl;
}

int main(int argc, char **argv) {
  if (boost::filesystem::exists(
          boost::filesystem::path("log4cplus.properties"))) {
    log4cplus::PropertyConfigurator::doConfigure("log4cplus.properties");
  } else {
    log4cplus::BasicConfigurator::doConfigure();
  }

  log4cplus::Logger logger = log4cplus::Logger::getInstance("ecs.main");

  if (argc < 3) {
    print_usage();
    exit(1);
  }

  string outfn("out.off");
  outfn = argv[argc - 1];

  ecs_Bbox_3 bb;
  bool bb_init = false;
  bool box_only = false;
  bool keep_all = false;
  bool crop_only = false;
  double mins[3];
  double maxs[3];

  int argi = 1;
  while (argv[argi][0] == '-') {
    string arg(argv[argi++]);
    if (arg == "-p") {
      string p(argv[argi++]);
      for (int j = 1; j <= 6; j++) {
        bb.apply[j - 1] =
            p.find(boost::lexical_cast<string>(j)) != string::npos;
      }
    } else if (arg == "-o") {
      outfn = argv[argi++];
    } else if (arg == "-c") {
      crop_only = true;
    } else if (arg == "-h") {
      print_usage();
      exit(0);
    } else if (arg == "-b") {
      for (int i = 0; i < 3; ++i) {
        mins[i] = boost::lexical_cast<double>(argv[i + argi]);
        maxs[i] = boost::lexical_cast<double>(argv[i + 3 + argi]);
      }
      bb.init(mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2]);
      bb_init = true;
      argi += 6;
      // box_only = true;
    } else if (arg == "-a") {
      keep_all = true;
    } else {
      argi--;
      break;
    }
  }

  vector<string> filenames;
  for (int i = argi; i < argc - 1; ++i) {
    filenames.push_back(argv[i]);
  }
  process_ecs(filenames, outfn, bb, bb_init, crop_only);
}

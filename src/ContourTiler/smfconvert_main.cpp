#include <stdio.h>
#include <string>
#include <cstring>
#include <ContourTiler/smfconvert.h>

using namespace SMF;
using namespace std;

void printUsage()
{
  printf("Usage:\n");
  printf("./smf2raw -smf2raw InputFileName OutputFileName\n");
  printf("./smf2raw -raw2smf InputFileName OutputFileName\n");
}

int main(int argc, char* argv[])
{
  if (argc != 4)
  {
    printUsage();
    return 0;
  }
  string infile(argv[2]);
  string outfile(argv[3]);
  if (strstr(argv[1], "smf2raw") != NULL)
  {
    smf2raw(infile, outfile);
    // smf2raw(argc, argv);
    return 0;
  }
  if (strstr(argv[1], "raw2smf") != NULL)
  {
    raw2smf(infile, outfile);
    // raw2smf(argc, argv);
    return 0;
  }
  printUsage();
  return 0;
}


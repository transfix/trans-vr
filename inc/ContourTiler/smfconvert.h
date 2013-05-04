#ifndef __SMF_H__
#define __SMF_H__

#define SMF_BEGIN_NAMESPACE namespace SMF {
#define SMF_END_NAMESPACE }

#include <string>

SMF_BEGIN_NAMESPACE

void smf2raw(std::string infile, std::string outfile);
void raw2smf(std::string infile, std::string outfile);

SMF_END_NAMESPACE

#endif

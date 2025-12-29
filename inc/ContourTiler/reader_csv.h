#ifndef __READER_CSV_H__
#define __READER_CSV_H__

#include "boost/tuple/tuple.hpp"

#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <fstream>
#include <sstream>

CONTOURTILER_BEGIN_NAMESPACE

boost::tuple<std::string, std::string, std::string>
dendrite_axon(const std::string &line, size_t axon_index) {
  using namespace std;

  typedef boost::tokenizer<boost::escaped_list_separator<char>> tokenizer;
  const int SYNAPSE_INDEX = 1;
  const int MAX_INDEX = axon_index;

  string synapse, dendrite, axon;
  tokenizer tok(line);
  int index = 0;
  for (tokenizer::iterator it = tok.begin();
       index <= MAX_INDEX && it != tok.end(); ++it, ++index) {
    if (index == SYNAPSE_INDEX) {
      synapse = dendrite = *it;
    } else if (index == axon_index)
      axon = *it;
  }

  std::cout << "adding synapse " << synapse << std::endl;
  if (!dendrite.empty())
    dendrite = dendrite.substr(0, dendrite.find('c'));
  try {
    int ind = boost::lexical_cast<int>(axon);
    stringstream ss;
    ss << "a" << setfill('0') << setw(3) << ind;
    axon = ss.str();
  } catch (boost::bad_lexical_cast &) {
    // has a letter on the end
    axon = "a" + axon;
  }
  return boost::make_tuple(synapse, dendrite, axon);
}

size_t axon_index(const std::string &line) {
  using namespace std;

  typedef boost::tokenizer<boost::escaped_list_separator<char>> tokenizer;

  tokenizer tok(line);
  int index = 0;
  for (tokenizer::iterator it = tok.begin(); it != tok.end(); ++it, ++index) {
    if (it->compare("axon") == 0)
      return index;
  }
  throw std::runtime_error("Axon index not found in csv file");
  return 0;
}

template <typename OutputIterator>
void read_synapses_csv(std::istream &in, OutputIterator out) {
  using namespace std;

  const int FIRST_LINE = 1;
  string line;

  getline(in, line); // read column headers
  size_t axon_idx = axon_index(line);

  while (!in.eof()) {
    getline(in, line);
    if (!in.eof()) {
      string synapse, dendrite, axon;
      boost::tie(synapse, dendrite, axon) = dendrite_axon(line, axon_idx);
      if (!synapse.empty() && !dendrite.empty() && !axon.empty()) {
        *out = boost::make_tuple(synapse, dendrite, axon);
        ++out;
      }
    }
  }
}

template <typename OutputIterator>
void read_synapses_csv(const std::string &filename, OutputIterator out) {
  std::ifstream in(filename.c_str());
  read_synapses_csv(in, out);
  in.close();
}

CONTOURTILER_END_NAMESPACE

#endif

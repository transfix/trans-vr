// Author: Justin Kinney
// Date: Feb 2009

#ifndef CF_HISTOGRAM_H
#define CF_HISTOGRAM_H 1

#include "common.h"
#include "reconstruct2contourtiler.h"

#include <iostream>

typedef std::vector<double> vec_d;
typedef std::vector<double>::iterator d_iterator;
typedef std::vector<double>::const_iterator c_d_iterator;

ContourFilter_BEGIN_NAMESPACE

    class Histogram {
private:
  Histogram &operator=(Histogram const &);
  Histogram(Histogram const &);
  static const int num_bins = 16;
  int N; // number of data points
  double min;
  double max;
  double sum;
  double sum2;
  int *count;
  double *bins;
  vec_d x;

public:
  void scan(const vec_d &my_path_intervals);
  ~Histogram(void);
  Histogram(void);
  void printHistogram(void);
  void printStats(void) const;
  void createHistogram(void);
  void update(double const &);
  void printStatistics(std::string str);
  double mean(void) const;
  double variance(void) const;

public:
  double getMax(void) const { return max; }
  double getSum(void) const { return sum; }
  int getN(void) const { return N; }
  void incrementCount(int i) {
    assert(i < 16);
    count[i]++;
  }

private:
  void setMin(double d) { min = d; }
  void setMax(double d) { max = d; }
  void setSum(double d) { sum = d; }
  double getMin(void) const { return min; }
};

ContourFilter_END_NAMESPACE

#endif

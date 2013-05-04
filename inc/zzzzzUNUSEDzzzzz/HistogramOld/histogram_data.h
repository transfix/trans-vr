#ifndef __HISTOGRAM_DATA_H__
#define __HISTOGRAM_DATA_H__

#include <vector>
#include <iostream>
#include <cassert>

class HistogramData
{
	public:
		HistogramData();
		HistogramData(const std::vector<double>& inputPoints, unsigned int numBins);
		~HistogramData();
		void debug();
		void save();
		void load();
		int width()
		{
			if (!_initialized)
			{
				return 0;
			}
			return (int)_bins.size();
		}
		unsigned int height()
		{
			if (!_initialized)
			{
				return 0;
			}
			return _binMax;
		}
		double getBin(int i)
		{
			if (!_initialized)
			{
				return 0;
			}
			if (i<0 || i>=width())
			{
				return 0;
			}
			return _bins[i];
		}
		double getBinNormalized(int i)
		{
			if (!_initialized)
			{
				return 0;
			}
			if (i<0 || i>=width())
			{
				return 0;
			}
			return _bins[i]/(double)_binMax;
		}
		void rebin(int newBins);
		double binToWidth(int bin);

	private:
		void add(double input);
		bool _initialized;
		double _inputMin, _inputMax;
		unsigned int _binMin, _binMax;
		std::vector<double> _inputPoints;
		std::vector< unsigned int > _bins;
		double _binWidth;
};

#endif

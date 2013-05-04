#include <cfloat>
#include <climits>
#include <fstream>
#include <Histogram/histogram_data.h>

HistogramData::HistogramData()
{
	_initialized = false;
	// find min and max
	_inputMin = DBL_MAX;
	_inputMax = DBL_MIN;
	_binMin = INT_MAX;
	_binMax = 0;
	_binWidth = 0;
	// zero out the bins
	_bins.clear();
}

HistogramData::HistogramData(const std::vector< double >& inputPoints, unsigned int numBins)
{
	// set initialized
	_initialized = true;
	// find min and max
	_inputMin = DBL_MAX;
	_inputMax = DBL_MIN;
	for (int i=0; i< inputPoints.size(); i++)
	{
		if (inputPoints[i] < _inputMin)
		{
			_inputMin = inputPoints[i];
		}
		else if (inputPoints[i] > _inputMax)
		{
			_inputMax = inputPoints[i];
		}
	}
	// copy points
	_inputPoints = inputPoints;
	// rebin
	rebin(numBins);
}

HistogramData::~HistogramData() {}

void HistogramData::add(double input)
{
	if (!_initialized)
	{
		return;
	}
	// find bin
	assert(input >= _inputMin);
	assert(input <= _inputMax);
	unsigned int binIndex = (unsigned int)((input - _inputMin)/_binWidth);
	_bins[binIndex]++;
}

void HistogramData::debug()
{
	if (!_initialized)
	{
		return;
	}
	std::cerr << "\n";
	for (int i=0; i<_bins.size(); i++)
	{
		for (int j=0; j<_bins[i]; j++)
		{
			std::cerr << "*";
		}
		std::cerr << "\n";
	}
	std::cerr << "\n";
}

void HistogramData::save()
{
	if (!_initialized)
	{
		return;
	}
	std::ofstream outfile("/ices/mstrange/cvs/TexMol/histogramdata.txt");
	outfile <<  _initialized << "\t"
			<<  _inputMin << "\t"
			<<  _inputMax << "\t"
			<< _binWidth << "\t";
	outfile << _inputPoints.size() << "\t";
	for (int i=0; i<_inputPoints.size(); i++)
	{
		double d = _inputPoints[i];
		outfile << d << "\t";
	}
	outfile.close();
}

void HistogramData::load()
{
	std::ifstream infile("/ices/mstrange/cvs/TexMol/histogramdata.txt");
	infile >>  _initialized
		   >>  _inputMin
		   >>  _inputMax
		   >> _binWidth;
	unsigned int nPoints;
	infile >> nPoints;
	_inputPoints.clear();
	for (int i=0; i<nPoints; i++)
	{
		double d;
		infile >> d;
		_inputPoints.push_back(d);
	}
	infile.close();
	// until later...
	rebin(1);
}

void HistogramData::rebin(int newBins)
{
	if (!_initialized)
	{
		return;
	}
	// determine widths
	double totalWidth = _inputMax - _inputMin;
	_binWidth = (totalWidth)/newBins;
	// zero out the bins
	_bins.clear();
	for (int i=0; i<newBins; i++)
	{
		_bins.push_back(0);
	}
	// add the points
	for (int i=0; i<_inputPoints.size(); i++)
	{
		add(_inputPoints[i]);
	}
	// find _binMin, _binMax
	_binMin = INT_MAX;
	_binMax = 0;
	for (int i=0; i<_bins.size(); i++)
	{
		if (_bins[i] < _binMin)
		{
			_binMin = _bins[i];
		}
		if (_bins[i] > _binMax)
		{
			_binMax = _bins[i];
		}
	}
}

double HistogramData::binToWidth(int bin)
{
	if (bin<0 || bin>=width())
	{
		return 0;
	}
	return _inputMin + (_binWidth * bin);
}

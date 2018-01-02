#include <iostream>
#include <string>
#include <algorithm>
#include <netcdf>
#include <boost/progress.hpp>
#include <blitz/array.h>

namespace nb {

typedef std::map<std::string, std::pair<std::string, size_t> > rpl_map;
typedef std::vector<netCDF::NcDim> dim_vec;
typedef std::vector<std::vector<int> > err_vec;
typedef std::vector<std::string> str_vec;


class Clock: public boost::progress_display
{
	clock_t t1,t2;
	public:
	Clock(const size_t x);
	~Clock();
};


// Note: template members easiest to define in header, not implementation file

/** \brief Wrapper containing a netCDF::NcVar as a member.

On instantiation, the record dimension and the netCDF ``start`` and ``count`` parameters for the variable are determined.
*/
template <class T, int n=4>
class Var
{
	protected:
	netCDF::NcVar var;
// 	blitz::TinyVector<int, n-1> shape;
	std::vector<size_t> start, count;
	std::vector<int> ndims;
	size_t _reclen;
	int _recdim;
	void error_loc(int t,int y,int x, err_vec &err)
	{
		std::vector<int> idx = {t, y, x};
		err.push_back(idx);
	};
	public:
	// if defined in nb.cpp, would need to be instantiated as, e.g.:
	// template class Var<double>;
  /**
     \param v The netCDF::NcVar.
  */
	Var(const netCDF::NcVar &v): var(v), start(n, 0), count(n, 1)
	{
		_reclen = 0;
		std::vector< netCDF::NcDim > dims = var.getDims();
		for (int i=0,j=0; i<dims.size(); ++i) {
			if (dims[i].isUnlimited()) {
				_reclen = dims[i].getSize();
				_recdim = i;
			} else {
				count[i] = dims[i].getSize();
				ndims.push_back(i);
			}
		}
	};

	size_t reclen() { return _reclen; }
  blitz::TinyVector<size_t, n> shape() {
    blitz::TinyVector<size_t, n> c(count.data());
    if (_reclen != 0)
      c[_recdim] = _reclen;
    return c;
  }

  blitz::Array<T, n> getFullArray()
  {
    blitz::Array<T, n> arr(shape());
    var.getVar(start, count, arr.data());
    return arr;
  }

  /**
     Returns an empty blitz::Array with the same shape as the underlying variable, but with the record dimension removed.
   */
	template<int m=n-1>
	blitz::Array<T, m> array()
	{
		blitz::TinyVector<int,m> s;
		for (int i=0; i<m; ++i) s[i] = count[ndims[i]];
		return blitz::Array<T,m>(s);
	};
	template<int m=n-1>
	void getVar(blitz::Array<T,m> &arr)
	{
		var.getVar(start, count, arr.data());
		++start[0];
	};
	// see in nb.cpp for instantiation of template
	dim_vec copy_dims(netCDF::NcFile*, const rpl_map&, const netCDF::NcVar *ul = NULL);
	dim_vec copy_dims(netCDF::NcFile*, const str_vec&, const netCDF::NcVar *ul = NULL);
	dim_vec copy_dims(netCDF::NcFile*, const std::vector<int> &, const netCDF::NcVar *ul = NULL);
};

/**
My own subclass of netCDF::NcFile.
*/
class File: public netCDF::NcFile
{
	typedef std::multimap<std::string, netCDF::NcGroupAtt> gatt_map;
	typedef std::multimap<std::string, netCDF::NcVar> var_map;

	public:
	File (const std::string &filename, netCDF::NcFile::FileMode mode = netCDF::NcFile::read):
		netCDF::NcFile(filename, mode) {};
	~File() { close(); }
	void vars();
	netCDF::NcVar unlimited(const netCDF::NcVar &);
	File* copy_atts(const std::string&);
};

}

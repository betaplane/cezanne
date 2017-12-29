#include "nb.h"
using namespace nb;

Clock::Clock(const size_t x): boost::progress_display(x) { t1 = std::clock(); };
Clock::~Clock()
{
	t2 = clock();
	float diff = ((float)t2-(float)t1)/CLOCKS_PER_SEC;
	std::cout << "this took " << diff << " seconds" << std::endl;
};


File* File::copy_atts(const std::string &path) {
	File *f = new File(path, netCDF::NcFile::replace);
	gatt_map g_atts = this->getAtts();

	for (gatt_map::iterator i=g_atts.begin(); i!=g_atts.end(); ++i) {
		netCDF::NcGroupAtt att = i->second;
		netCDF::NcType type = att.getType();
		size_t len = att.getAttLength();
		size_t bytes = type.getSize();
		void *data = malloc(len*bytes);
		att.getValues(data);
		f->putAtt(i->first, type, len, data);
		free(data);
	}
	return f;
}


void ul_var(netCDF::NcFile *f, const netCDF::NcVar *v, const netCDF::NcDim &d)
{
	std::string units;
	size_t n = v->getDim(0).getSize();
	v->getAtt("units").getValues(units);
	std::vector<size_t> start(1,0), count(1,n);
	blitz::Array<float,1> t(n);
	v->getVar(t.data());
	netCDF::NcVar time = f->addVar(v->getName(), v->getType(), d);
	time.putAtt("units", units);
	time.putVar(start, count, t.data());
};


// Note: check if dims exist first (in case multiple variables with different dims are being added)
template<class T, int n>
dim_vec Var<T,n>::copy_dims(netCDF::NcFile *f, const rpl_map &replace, const netCDF::NcVar *ul)
{
	dim_vec out, dims = var.getDims();
	size_t s;
	std::string name;
	rpl_map::const_iterator it;

	for (auto dim = dims.begin(); dim != dims.end(); dim++) {
		netCDF::NcDim new_dim;
		name = dim->getName();
		it = replace.find(name);
		if (dim->isUnlimited()) {
			new_dim = f->addDim(name);
			if (ul) ul_var(f, ul, new_dim);
			std::cout << name << ", unlimited" << std::endl;
		} else {
			if (it==replace.end()) {
				s = dim->getSize();
				new_dim = f->addDim(name, s);
			} else {
				s = (it->second).second;
				new_dim = f->addDim((it->second).first, s);
			}
			std::cout << name << ", " << s << std::endl;
		}
		out.push_back( new_dim );
	}
	return out;
};

template<class T, int n>
dim_vec Var<T,n>::copy_dims(netCDF::NcFile *f, const str_vec &drop, const netCDF::NcVar *ul)
{
	dim_vec out, dims = var.getDims();
	size_t s;
	std::string name;
	str_vec::const_iterator it;

	for (auto dim = dims.begin(); dim != dims.end(); dim++) {
		name = dim->getName();
		it = std::find(drop.begin(), drop.end(), name);
		if (it!=drop.end()) continue;
		netCDF::NcDim new_dim;

		if (dim->isUnlimited()) {
			new_dim = f->addDim(name);
			if (ul) ul_var(f, ul, new_dim);
			std::cout << name << ", unlimited" << std::endl;
		} else {
			s = dim->getSize();
			new_dim = f->addDim(name, s);
			std::cout << name << ", " << s << std::endl;
		}
		out.push_back( new_dim );
	}
	return out;
};


template<class T, int n>
dim_vec Var<T,n>::copy_dims(netCDF::NcFile *f, const std::vector<int> &take, const netCDF::NcVar *ul)
{
	size_t s;
	std::string name;
	dim_vec dims, out;

	for (auto i = take.begin(); i != take.end(); i++) {
		netCDF::NcDim new_dim, dim = var.getDim(*i);
		name = dim.getName();
		dims.push_back( dim );

		if (dim.isUnlimited()) {
			new_dim = f->addDim(name);
			if (ul) ul_var(f, ul, new_dim);
			std::cout << name << ", unlimited" << std::endl;
		} else {
			s = dim.getSize();
			new_dim = f->addDim(name, s);
			std::cout << name << ", " << s << std::endl;
		}
		out.push_back( new_dim );
	}
	return out;
};

// if actually used, needs to be instantiated here for given type:
template dim_vec Var<double>::copy_dims(netCDF::NcFile*, const rpl_map&, const netCDF::NcVar*);
template dim_vec Var<float>::copy_dims(netCDF::NcFile*, const rpl_map&, const netCDF::NcVar*);
template dim_vec Var<double>::copy_dims(netCDF::NcFile*, const str_vec&, const netCDF::NcVar*);
template dim_vec Var<float>::copy_dims(netCDF::NcFile*, const str_vec&, const netCDF::NcVar*);
template dim_vec Var<double>::copy_dims(netCDF::NcFile*, const std::vector<int>&, const netCDF::NcVar*);
template dim_vec Var<float>::copy_dims(netCDF::NcFile*, const std::vector<int>&, const netCDF::NcVar*);



void File::vars() {
	typedef std::map<std::string, netCDF::NcVarAtt> att_map;

	var_map vars = this->getVars();
	for (var_map::iterator i=vars.begin(); i!=vars.end(); ++i)
	{
		std::cout << (*i).first << std::endl;
		att_map atts = (*i).second.getAtts();
		for (att_map::iterator j=atts.begin(); j!=atts.end(); ++j)
		{
			std::cout << "    " << j->first << std::endl;
		}
	}
};


CXX = g++
BASE = /sata1_ceazalabs/arno/HPC/uvHome/miniconda3/envs/cxx

CPPFLAGS = -I$(BASE)/include -pthread -std=c++0x
# see libnetcdf-cxx.settings in lib directory for compiler options
LDFLAGS = -Wl,-rpath,$(BASE)/lib,-rpath,$(CURDIR) -L$(BASE)/lib -lnetcdf_c++4 -lnetcdf -lblitz

lib: libnb.so

test: test.cpp $(lib)
	$(CXX) -o $@ $^ $(CPPFLAGS) $(LDFLAGS) -L$(CURDIR) -lnb

libnb.so: libnb.o
	$(CXX) -o $@ $^ $(LDFLAGS) -shared

libnb.o: nb.cpp
	$(CXX) -fpic -o $@ -c $^ $(CPPFLAGS) $(LDFLAGS)

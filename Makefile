CXX = g++
BASE = /sata1_ceazalabs/arno/HPC/uvHome/miniconda3/envs/cxx

CPPFLAGS = -I$(BASE)/include -pthread -std=c++0x
# see libnetcdf-cxx.settings in lib directory for compiler options
LDFLAGS = -Wl,-rpath,$(BASE)/lib -L$(BASE)/lib -lnetcdf_c++4 -lnetcdf -lblitz

lib: libnb.so

test: test.cpp
	$(CXX) -o $@ $^ $(CPPFLAGS) $(LDFLAGS)

libnb.so: libnb.o
	$(CXX) -o $@ $^ $(LDFLAGS) -shared

libnb.o: nb.cpp
	$(CXX) -fpic -o $@ -c $^ $(CPPFLAGS) $(LDFLAGS)

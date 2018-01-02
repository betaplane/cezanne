#include "nb.h"

class nbFile: public nb::File
{
public:
  nbFile(const std::string &name): nb::File(name) {};
}

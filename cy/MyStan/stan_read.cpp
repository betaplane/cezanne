#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

std::vector<std::vector<float> > read_var(const std::string &file_name, const std::string &var_name) {
  std::ifstream ifs(file_name.c_str());
  std::string line;
  std::vector<int> columns;
  std::vector<std::vector<float> > data;
  int l = var_name.length();
  bool init1 = false;
  bool init2 = false;

  if ( !ifs.is_open() )
    std::cout << "Could not open file\n";
  else {
    while (std::getline(ifs, line)) {
      std::istringstream ss(line);
      int i = 0;
      std::string s;
      while (std::getline(ss, s, ',')) {
        if (s == line) break;
        if ((s.substr(0, l) == var_name) && (s[l] == '.'))
          columns.push_back(i);
        else if (init1) init2 = true;
        i++;
      }
      if (columns.size() > 0) init1 = true;
      if (init2) break;
    }

    while (std::getline(ifs, line)) {
      std::istringstream ss(line);
      std::string s;
      int i = 0;
      int j = 0;
      std::vector<float> row;
      while (std::getline(ss, s, ',')) {
        if (i++ == columns[j]) {
          row.push_back(std::stod(s));
          j++;
        }
        if (j == columns.size()) break;
      }
      data.push_back(row);
    }
  }
  return data;
}

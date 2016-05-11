//
// Created by tsv on 29.04.16.
//

#ifndef CUDA_VTK_SOLVER_H
#define CUDA_VTK_SOLVER_H

#include <vector>
#include <string>
#include <iostream>

class Solver {
 private:
  bool done = false;
  bool is_y_current = true;
  bool is_x_consistent = false;

  size_t N;
  size_t size;

  size_t threads = 1024;
  size_t blocks;
  size_t grids;

  float *d_y_in;
  float *d_y_out;

  float x0 = 0.f;
  float x1 = 1.f;

  float t = 0.f;
  float t_end = 1.f;

  float h;
  float tau = 1e-3f;

  void alloc_mem();
  void free_mem();

  void fill_x();

  std::vector<float> x;
  std::vector<float> y;

 protected:
  virtual void next_step() = 0;

  void recalc_h();
  void copy_from_device();

  float *get_d_y_in();
  float *get_d_y_out();

  const size_t &get_threads();
  const size_t &get_blocks();
  const size_t &get_grids();

  std::ofstream get_file(const std::string &file_path, const std::string &file_name);

 public:
  Solver(size_t N = 1024);
  virtual ~Solver();

  void step();

  virtual void save(const std::string &file_path, const std::string &file_name);
  bool is_done();

  const std::vector<float> &get_x();
  const std::vector<float> &get_y();
  const float &get_t();
  const float &get_h();
  const float &get_tau();
  const size_t &get_N();

  void set_x0(const float &new_x0);
  void set_x1(const float &new_x1);
  void set_y(const std::vector<float> &new_y);
  void set_tau(const float &new_tau);
  void set_h(const float &new_h);
  void set_N(const size_t &new_N);
  void set_t_end(const float &new_t_end);
};

class Cuda_exception: public std::exception {
 protected:
  std::string msg;

 public:
  explicit Cuda_exception(const char *message)
      : msg(message) {
  }

  explicit Cuda_exception(const std::string message)
      : msg(message) {
  }

  virtual ~Cuda_exception() {
  }

  virtual const char *what() const throw() {
    return msg.c_str();
  }
};

#endif //CUDA_VTK_SOLVER_H

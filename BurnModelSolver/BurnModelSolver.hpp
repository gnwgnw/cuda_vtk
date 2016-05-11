//
// Created by tsv on 06.05.16.
//

#ifndef CUDA_VTK_BURNMODELSOLVER_H
#define CUDA_VTK_BURNMODELSOLVER_H

#include "Solver/Solver.hpp"
#include "Params.hpp"


class BurnModelSolver: public Solver {
 private:
  Params params;
  Params *d_params = nullptr;

 protected:
  virtual void next_step() override;

 public:
  BurnModelSolver(size_t N = 1024)
      : Solver(N) {
  }

  virtual ~BurnModelSolver();

  virtual void save(const std::string &file_path, const std::string &file_name) override;

  const Params &get_params();
  void set_params(const Params &new_params);
};


#endif //CUDA_VTK_BURNMODELSOLVER_H

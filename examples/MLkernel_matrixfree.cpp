/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The Regents of
 * the University of California, through Lawrence Berkeley National Laboratory
 * (subject to receipt of any required approvals from the U.S. Dept. of Energy).
 * All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Technology Transfer Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As such, the
 * U.S. Government has been granted for itself and others acting on its behalf a
 * paid-up, nonexclusive, irrevocable, worldwide license in the Software to
 * reproduce, prepare derivative works, and perform publicly and display
 * publicly. Beginning five (5) years after the date permission to assert
 * copyright is obtained from the U.S. Department of Energy, and subject to any
 * subsequent five (5) year renewals, the U.S. Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive, irrevocable,
 * worldwide license in the Software to reproduce, prepare derivative works,
 * distribute copies to the public, perform publicly and display publicly, and
 * to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 * Division).
 *
 */
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "HSS/HSSMatrix.hpp"
#include "misc/TaskTimer.hpp"

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;

#define ERROR_TOLERANCE 1e2

const int kmeans_max_it = 1000;
random_device rd;
double r;
mt19937 generator(rd());

inline double dist2(double *x, double *y, int d) {
  double k = 0.;
  for (int i = 0; i < d; i++)
    k += pow(x[i] - y[i], 2.);
  return k;
}

inline double dist(double *x, double *y, int d) { return sqrt(dist2(x, y, d)); }

inline double norm1(double *x, double *y, int d) {
  double k = 0.;
  for (int i = 0; i < d; i++)
    k += fabs(x[i] - y[i]);
  return k;
}

inline double Gauss_kernel(double *x, double *y, int d, double h) {
  return exp(-dist2(x, y, d) / (2. * h * h));
}

inline double Laplace_kernel(double *x, double *y, int d, double h) {
  return exp(-norm1(x, y, d) / h);
}

inline int *kmeans_start_random(int n, int k) {
  uniform_int_distribution<int> uniform_random(0, n - 1);
  int *ind_centers = new int[k];
  for (int i = 0; i < k; i++) {
    ind_centers[i] = uniform_random(generator);
  }
  return ind_centers;
}

// 3 more start sampling methods for the case k == 2

int *kmeans_start_random_dist_maximized(int n, double *p, int d) {
  constexpr size_t k = 2;

  uniform_int_distribution<int> uniform_random(0, n - 1);
  const auto t = uniform_random(generator);
  // compute probabilities
  double *cur_dist = new double[n];
  for (int i = 0; i < n; i++) {
    cur_dist[i] = dist2(&p[i * d], &p[t * d], d);
  }

  std::discrete_distribution<int> random_center(&cur_dist[0], &cur_dist[n]);

  delete[] cur_dist;

  int *ind_centers = new int[k];
  ind_centers[0] = t;
  ind_centers[1] = random_center(generator);
  return ind_centers;
}

int *kmeans_start_dist_maximized(int n, double *p, int d) {
  constexpr size_t k = 2;

  // find cenroid
  double centroid[d];

  for (int i = 0; i < d; i++) {
    centroid[i] = 0;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      centroid[j] += p[i * d + j];
    }
  }

  for (int j = 0; j < d; j++)
    centroid[j] /= n;

  // find farthest point from centroid
  int first_index = 0;
  double max_dist = -1;

  for (int i = 0; i < n; i++) {
    double dd = dist(&p[i * d], centroid, d);
    if (dd > max_dist) {
      max_dist = dd;
      first_index = i;
    }
  }

  // find fathest point from the first point
  int second_index = 0;
  max_dist = -1;
  for (int i = 0; i < n; i++) {
    double dd = dist(&p[i * d], &p[first_index * d], d);
    if (dd > max_dist) {
      max_dist = dd;
      second_index = i;
    }
  }
  int *ind_centers = new int[k];
  ind_centers[0] = first_index;
  ind_centers[1] = second_index;
  return ind_centers;
}

inline int *kmeans_start_fixed(int n, double *p, int d) {
  int *ind_centers = new int[2];
  ind_centers[0] = 0;
  ind_centers[1] = n - 1;
  return ind_centers;
}

void k_means(int k, double *p, int n, int d, int *nc) {
  double **center = new double *[k];

  int *ind_centers = NULL;

  constexpr int kmeans_option = 2;

  switch (kmeans_option) {
  case 1:
    ind_centers = kmeans_start_random(n, k);
    break;
  case 2:
    ind_centers = kmeans_start_random_dist_maximized(n, p, d);
    break;
  case 3:
    ind_centers = kmeans_start_dist_maximized(n, p, d);
    break;
  case 4:
    ind_centers = kmeans_start_fixed(n, p, d);
    break;
  }

  for (int c = 0; c < k; c++) {
    center[c] = new double[d];
    for (int j = 0; j < d; j++)
      center[c][j] = p[ind_centers[c] * d + j];
  }

  int iter = 0;
  bool changes = true;
  int *cluster = new int[n];
  for (int i = 0; i < n; i++) {
    cluster[i] = 0;
  }

  while ((changes == true) and (iter < kmeans_max_it)) {
    // for each point, find the closest cluster center
    changes = false;
    for (int i = 0; i < n; i++) {
      int old_cluster = cluster[i];
      double min_dist = dist(&p[i * d], center[0], d);
      cluster[i] = 0;
      for (int c = 1; c < k; c++) {
        double dd = dist(&p[i * d], center[c], d);
        if (dd <= min_dist) {
          min_dist = dd;
          cluster[i] = c;
        }
      }
      if (old_cluster != cluster[i]) {
        changes = true;
      }
    }

    for (int c = 0; c < k; c++) {
      nc[c] = 0;
      for (int j = 0; j < d; j++)
        center[c][j] = 0.;
    }
    for (int i = 0; i < n; i++) {
      auto c = cluster[i];
      nc[c]++;
      for (int j = 0; j < d; j++)
        center[c][j] += p[i * d + j];
    }
    for (int c = 0; c < k; c++)
      for (int j = 0; j < d; j++)
        center[c][j] /= nc[c];
    iter++;
  }

  int *ci = new int[k];
  for (int c = 0; c < k; c++)
    ci[c] = 0;
  double *p_perm = new double[n * d];
  int row = 0;
  for (int c = 0; c < k; c++) {
    for (int j = 0; j < nc[c]; j++) {
      while (cluster[ci[c]] != c)
        ci[c]++;
      for (int l = 0; l < d; l++)
        p_perm[l + row * d] = p[l + ci[c] * d];
      ci[c]++;
      row++;
    }
  }
  copy(p_perm, p_perm + n * d, p);
  delete[] p_perm;
  delete[] ci;

  for (int i = 0; i < k; i++)
    delete[] center[i];
  delete[] center;
  delete[] cluster;
  delete[] ind_centers;
}

void recursive_2_means(double *p, int n, int d, int cluster_size,
                       HSSPartitionTree &tree) {
  if (n < cluster_size)
    return;
  auto nc = new int[2];
  k_means(2, p, n, d, nc);
  if (nc[0] == 0 || nc[1] == 0)
    return;
  tree.c.resize(2);
  tree.c[0].size = nc[0];
  tree.c[1].size = nc[1];
  recursive_2_means(p, nc[0], d, cluster_size, tree.c[0]);
  recursive_2_means(p + nc[0] * d, nc[1], d, cluster_size, tree.c[1]);
  delete[] nc;
}

void kd_partition(double *p, int n, int d, int *nc) {
  // find coordinate of the most spread
  double *maxes = new double[d];
  double *mins = new double[d];

  for (int j = 0; j < d; ++j) {
    maxes[j] = p[j];
    mins[j] = p[j];
  }

  for (int i = 1; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      if (p[i * d + j] > maxes[j]) {
        maxes[j] = p[i * d + j];
      }
      if (p[i * d + j] > mins[j]) {
        mins[j] = p[i * d + j];
      }
    }
  }
  double max_var = maxes[0] - mins[0];
  int dim = 0;
  for (int j = 0; j < d; ++j) {
    if (maxes[j] - mins[j] > max_var) {
      max_var = maxes[j] - mins[j];
      dim = j;
    }
  }

  // find the mean
  double mean_value = 0.;
  for (int i = 0; i < n; ++i) {
    mean_value += p[i * d + dim];
  }
  mean_value /= n;

  // split the data
  int *cluster = new int[n];
  for (int i = 0; i < n; i++) {
    cluster[i] = 0;
  }
  nc[0] = 0;
  nc[1] = 0;
  for (int i = 0; i < n; ++i) {
    if (p[d * i + dim] > mean_value) {
      cluster[i] = 1;
      nc[1] += 1;
    } else {
      nc[0] += 1;
    }
  }

  // permute the data

  int *ci = new int[2];
  for (int c = 0; c < 2; c++)
    ci[c] = 0;
  double *p_perm = new double[n * d];
  int row = 0;
  for (int c = 0; c < 2; c++) {
    for (int j = 0; j < nc[c]; j++) {
      while (cluster[ci[c]] != c)
        ci[c]++;
      for (int l = 0; l < d; l++)
        p_perm[l + row * d] = p[l + ci[c] * d];
      ci[c]++;
      row++;
    }
  }

  copy(p_perm, p_perm + n * d, p);
  delete[] p_perm;
  delete[] ci;
  delete[] maxes;
  delete[] mins;
  delete[] cluster;
}

void recursive_kd(double *p, int n, int d, int cluster_size,
                  HSSPartitionTree &tree) {
  if (n < cluster_size)
    return;
  auto nc = new int[2];
  kd_partition(p, n, d, nc);
  if (nc[0] == 0 || nc[1] == 0)
    return;
  tree.c.resize(2);
  tree.c[0].size = nc[0];
  tree.c[1].size = nc[1];
  recursive_kd(p, nc[0], d, cluster_size, tree.c[0]);
  recursive_kd(p + nc[0] * d, nc[1], d, cluster_size, tree.c[1]);
  delete[] nc;
}

class Kernel {
  using DenseM_t = DenseMatrix<double>;
  using DenseMW_t = DenseMatrixWrapper<double>;

public:
  vector<double> _data;
  int _d = 0;
  int _n = 0;
  double _h = 0.;
  double _l = 0.;
  Kernel() = default;
  Kernel(vector<double> data, int d, double h, double l)
    : _data(std::move(data)), _d(d), _n(_data.size() / _d), _h(h), _l(l) {
    assert(_n * _d == _data.size());
  }

  void operator()(const vector<size_t> &I, const vector<size_t> &J,
                  DenseM_t &B) {
    assert(I.size() == B.rows() && J.size() == B.cols());
    for (size_t j = 0; j < J.size(); j++) {
      for (size_t i = 0; i < I.size(); i++) {
        B(i, j) = Gauss_kernel(&_data[I[i] * _d], &_data[J[j] * _d], _d, _h);
        if (I[i] == J[j]){
          B(i,j) += _l;
        }
      }
    }
  }

  void times(DenseM_t &Rr, DenseM_t &Sr) {
    assert(Rr.rows() == _n);
    Sr.zero();
    const size_t B = 64;
    DenseM_t Asub(B, B);
#pragma omp parallel for firstprivate(Asub) schedule(dynamic)
    for (size_t c = 0; c < _n; c += B) {
      // loop over blocks of A
      for (size_t r = 0; r < _n; r += B) {
        const int Br = std::min(B, _n - r);
        const int Bc = std::min(B, _n - c);
        // construct a block of A
        for (size_t j = 0; j < Bc; j++) {
          for (size_t i = 0; i < Br; i++) {
            Asub(i, j) = Gauss_kernel
              (&_data[(r + i) * _d], &_data[(c + j) * _d], _d, _h);
          }
          if (r==c) Asub(j, j) += _l;
        }
        DenseMW_t Ablock(Br, Bc, Asub, 0, 0);
        // Rblock is a subblock of Rr of dimension Bc x Rr.cols(),
        // starting at position c,0 in Rr
        DenseMW_t Rblock(Bc, Rr.cols(), Rr, c, 0);
        DenseMW_t Sblock(Br, Sr.cols(), Sr, r, 0);
        // multiply block of A with a row-block of Rr and add result to Sr
        gemm(Trans::N, Trans::N, 1., Ablock, Rblock, 1., Sblock);
      }
    }

    // DenseM_t K(_n, _n);
    // for (size_t c=0; c<_n; c++)
    //   for (size_t r=0; r<_n; r++)
    //     K(r, c) = Gauss_kernel(&_data[r * _d], &_data[c * _d], _d, _h);
    // DenseM_t Stest(Sr.rows(), Sr.cols());
    // gemm(Trans::N, Trans::N, 1., K, Rr, 0., Stest);
    // Stest.scaled_add(-1., Sr);
    // std::cout << " ||Sr-Stest|| = " << Stest.norm() << std::endl;
  }

  void operator()(DenseM_t &Rr, DenseM_t &Rc, DenseM_t &Sr, DenseM_t &Sc) {
    times(Rr, Sr);
    //times(Rc, Sc);
    Sc.copy(Sr);
  }
};

int main(int argc, char *argv[]) {
  string filename("smalltest.dat");
  int d = 2;
  int reorder = 0;
  double h = 3.;
  double lambda = 0.;
  int kernel = 1; // Gaussian=1, Laplace=2

  cout << "# usage: ./ML_kernel file d h kernel(1=Gauss,2=Laplace) "
          "reorder(0=natural,1=recursive 2-means)"
       << endl;
  if (argc > 1)
    filename = string(argv[1]);
  if (argc > 2)
    d = stoi(argv[2]);
  if (argc > 3)
    h = stof(argv[3]);
  if (argc > 4)
    kernel = stoi(argv[4]);
  if (argc > 5)
    reorder = stoi(argv[5]);
  if (argc > 6)
    lambda = stof(argv[6]);
  cout << "# data dimension = " << d << endl;
  cout << "# kernel h = " << h << endl;
  cout << "# kernel type = " << ((kernel == 1) ? "Gauss" : "Laplace") << endl;
  cout << "# reordering/clustering = "
       << ((reorder == 0) ? "natural"
                          : ((reorder == 1) ? "recursive 2-means" : "kd"))
       << endl;
  cout << "# lambda = " << lambda << endl;
  HSSOptions<double> hss_opts;
  hss_opts.set_verbose(true);
  hss_opts.set_from_command_line(argc, argv);

  vector<double> data;
  {
    ifstream f(filename);
    string l;
    while (getline(f, l)) {
      istringstream sl(l);
      string s;
      while (getline(sl, s, ','))
        data.push_back(stod(s));
    }
    f.close();
  }

  int n = data.size() / d;
  cout << "# matrix size = " << n << " x " << d << endl;

  HSSPartitionTree cluster_tree;
  cluster_tree.size = n;
  int cluster_size = hss_opts.leaf_size();
  if (reorder == 1) {
    cout << " 2 means" << endl;
    recursive_2_means(data.data(), n, d, cluster_size, cluster_tree);
  } else if (reorder == 2) {
    recursive_kd(data.data(), n, d, cluster_size, cluster_tree);
    cout << "kd " << endl;
  }
  cout << "starting HSS compression .. " << endl;

  HSSMatrix<double> K;
  if (reorder != 0)
    K = HSSMatrix<double>(cluster_tree, hss_opts);
  else
    K = HSSMatrix<double>(n, n, hss_opts);

  TaskTimer::t_begin = GET_TIME_NOW();
  TaskTimer timer(string("compression"), 1);
  timer.start();
  Kernel kernel_matrix(data, d, h, lambda);
  K.compress(kernel_matrix, kernel_matrix, hss_opts);
  cout << "# compression time = " << timer.elapsed() << endl;

  if (K.is_compressed()) {
    cout << "# created K matrix of dimension " << K.rows() << " x " << K.cols()
         << " with " << K.levels() << " levels" << endl;
    cout << "# compression succeeded!" << endl;
  } else {
    cout << "# compression failed!!!!!!!!" << endl;
    return 1;
  }
  cout << "# rank(K) = " << K.rank() << endl;
  cout << "# memory(K) = " << K.memory() / 1e6 << " MB, " << endl;
  return 0;
}

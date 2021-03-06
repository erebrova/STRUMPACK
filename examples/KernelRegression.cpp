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

const int kmeans_max_it = 100;
random_device rd;
double r;
mt19937 generator(rd());

inline double dist2(double* x, double* y, int d) {
  double k = 0.;
  for (int i = 0; i < d; i++) k += pow(x[i] - y[i], 2.);
  return k;
}

inline double dist(double* x, double* y, int d) { return sqrt(dist2(x, y, d)); }

inline double norm1(double* x, double* y, int d) {
  double k = 0.;
  for (int i = 0; i < d; i++) k += fabs(x[i] - y[i]);
  return k;
}

inline double Gauss_kernel(double* x, double* y, int d, double h) {
  return exp(-dist2(x, y, d) / (2 * h * h));
}

inline double Laplace_kernel(double* x, double* y, int d, double h) {
  return exp(-norm1(x, y, d) / h);
}

inline int* kmeans_start_random(int n, int k) {
  uniform_int_distribution<int> uniform_random(0, n - 1);
  int* ind_centers = new int[k];
  for (int i = 0; i < k; i++) {
    ind_centers[i] = uniform_random(generator);
  }
  return ind_centers;
}

// 3 more start sampling methods for the case k == 2

int* kmeans_start_random_dist_maximized(int n, double* p, int d) {
  constexpr size_t k = 2;

  uniform_int_distribution<int> uniform_random(0, n - 1);
  const auto t = uniform_random(generator);
  // compute probabilities
  double* cur_dist = new double[n];
  for (int i = 0; i < n; i++) {
    cur_dist[i] = dist2(&p[i * d], &p[t * d], d);
  }

  std::discrete_distribution<int> random_center(&cur_dist[0], &cur_dist[n]);

  delete[] cur_dist;

  int* ind_centers = new int[k];
  ind_centers[0] = t;
  ind_centers[1] = random_center(generator);
  return ind_centers;
}

// for k = 2 only
int* kmeans_start_dist_maximized(int n, double* p, int d) {
  constexpr size_t k = 2;

  // find centroid
  double centroid[d];

  for (int i = 0; i < d; i++) {
    centroid[i] = 0;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      centroid[j] += p[i * d + j];
    }
  }

  for (int j = 0; j < d; j++) centroid[j] /= n;

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
  // find fathest point from the firsth point
  int second_index = 0;
  max_dist = -1;
  for (int i = 0; i < n; i++) {
    double dd = dist(&p[i * d], &p[first_index * d], d);
    if (dd > max_dist) {
      max_dist = dd;
      second_index = i;
    }
  }
  int* ind_centers = new int[k];
  ind_centers[0] = first_index;
  ind_centers[1] = second_index;
  return ind_centers;
}

inline int* kmeans_start_fixed(int n, double* p, int d) {
  int* ind_centers = new int[2];
  ind_centers[0] = 0;
  ind_centers[1] = n - 1;
  return ind_centers;
}

void k_means(int k, double* p, int n, int d, int* nc, double* labels) {
  double** center = new double*[k];

  int* ind_centers = NULL;

  constexpr int kmeans_options = 2;
  switch (kmeans_options) {
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
    for (int j = 0; j < d; j++) center[c][j] = p[ind_centers[c] * d + j];
  }

  int iter = 0;
  bool changes = true;
  int* cluster = new int[n];
  for (int i = 0; i < n; i++) {
    cluster[i] = 0;
  }

  while ((changes == true) and (iter < kmeans_max_it)) {
    // for each point, find the closest cluster center
    changes = false;
    for (int i = 0; i < n; i++) {
      double min_dist = dist(&p[i * d], center[0], d);
      cluster[i] = 0;
      for (int c = 1; c < k; c++) {
        double dd = dist(&p[i * d], center[c], d);
        if (dd <= min_dist) {
          min_dist = dd;
          if (c != cluster[i]) {
            changes = true;
          }
          cluster[i] = c;
        }
      }
    }

    for (int c = 0; c < k; c++) {
      nc[c] = 0;
      for (int j = 0; j < d; j++) center[c][j] = 0.;
    }
    for (int i = 0; i < n; i++) {
      auto c = cluster[i];
      nc[c]++;
      for (int j = 0; j < d; j++) center[c][j] += p[i * d + j];
    }
    for (int c = 0; c < k; c++)
      for (int j = 0; j < d; j++) center[c][j] /= nc[c];
    iter++;
  }

  int* ci = new int[k];
  for (int c = 0; c < k; c++) ci[c] = 0;
  double* p_perm = new double[n * d];
  double* labels_perm = new double[n];
  int row = 0;
  for (int c = 0; c < k; c++) {
    for (int j = 0; j < nc[c]; j++) {
      while (cluster[ci[c]] != c) ci[c]++;
      for (int l = 0; l < d; l++) p_perm[l + row * d] = p[l + ci[c] * d];
      labels_perm[row] = labels[ci[c]];
      ci[c]++;
      row++;
    }
  }
  copy(p_perm, p_perm + n * d, p);
  copy(labels_perm, labels_perm + n, labels);
  delete[] p_perm;
  delete[] labels_perm;
  delete[] ci;

  for (int i = 0; i < k; i++) delete[] center[i];
  delete[] center;
  delete[] cluster;
  delete[] ind_centers;
}

void recursive_2_means(double* p, int n, int d, int cluster_size,
                       HSSPartitionTree& tree, double* labels) {
  if (n < cluster_size) return;
  auto nc = new int[2];
  k_means(2, p, n, d, nc, labels);
  if (nc[0] == 0 || nc[1] == 0) return;
  tree.c.resize(2);
  tree.c[0].size = nc[0];
  tree.c[1].size = nc[1];
  recursive_2_means(p, nc[0], d, cluster_size, tree.c[0], labels);
  recursive_2_means(p + nc[0] * d, nc[1], d, cluster_size, tree.c[1],
                    labels + nc[0]);
  delete[] nc;
}

void kd_partition(double* p, int n, int d, int* nc, double* labels) {
  // find coordinate of the most spread
  double* maxes = new double[d];
  double* mins = new double[d];

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
  int* cluster = new int[n];
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

  int* ci = new int[2];
  for (int c = 0; c < 2; c++) ci[c] = 0;
  double* p_perm = new double[n * d];
  double* labels_perm = new double[n];
  int row = 0;
  for (int c = 0; c < 2; c++) {
    for (int j = 0; j < nc[c]; j++) {
      while (cluster[ci[c]] != c) ci[c]++;
      for (int l = 0; l < d; l++) p_perm[l + row * d] = p[l + ci[c] * d];
      labels_perm[row] = labels[ci[c]];
      ci[c]++;
      row++;
    }
  }

  copy(p_perm, p_perm + n * d, p);
  copy(labels_perm, labels_perm + n, labels);
  delete[] p_perm;
  delete[] labels_perm;
  delete[] ci;
  delete[] maxes;
  delete[] mins;
  delete[] cluster;
}

void recursive_kd(double* p, int n, int d, int cluster_size,
                  HSSPartitionTree& tree, double* labels) {
  if (n < cluster_size) return;
  auto nc = new int[2];
  kd_partition(p, n, d, nc, labels);
  if (nc[0] == 0 || nc[1] == 0) return;
  tree.c.resize(2);
  tree.c[0].size = nc[0];
  tree.c[1].size = nc[1];
  recursive_kd(p, nc[0], d, cluster_size, tree.c[0], labels);
  recursive_kd(p + nc[0] * d, nc[1], d, cluster_size, tree.c[1],
               labels + nc[0]);
  delete[] nc;
}

vector<double> write_from_file(int samples_size, string filename) {
  vector<double> data;
  ifstream f(filename);
  string l;
  int count = 0;
  while (getline(f, l) && count < samples_size) {
    istringstream sl(l);
    string s;
    while (getline(sl, s, ',')) data.push_back(stod(s));
    count++;
  }
  // cout << count << endl;
  return data;
}

int main(int argc, char* argv[]) {
  string filename("smalltest.dat");
  int d = 2;
  int reorder = 0;
  double h = 3.;
  double lambda = 1.;
  int kernel = 1;  // Gaussian=1, Laplace=2
  int samples_size=1;

  cout <<  endl << "# usage: ./KernelRegression file d h kernel(1=Gauss,2=Laplace) lambda "
          "reorder(0=natural,1=recursive 2-means) samples_size"
       << endl;
  if (argc > 1) filename = string(argv[1]);
  if (argc > 2) d = stoi(argv[2]);
  if (argc > 3) h = stof(argv[3]);
  if (argc > 4) kernel = stoi(argv[4]);
  if (argc > 5) reorder = stoi(argv[5]);
  if (argc > 6) lambda = stof(argv[6]);
  if (argc > 7) samples_size = stof(argv[7]);
  cout << "# data dimension = " << d << endl;
  cout << "# kernel h = " << h << endl;
  cout << "# kernel type = " << ((kernel == 1) ? "Gauss" : "Laplace") << endl;
  cout << "# reordering/clustering = "
       << ((reorder == 0) ? "natural"
                          : ((reorder == 1) ? "recursive 2-means" : "kd"))
       << endl;
  cout << "# lambda = " << lambda << endl;
  cout << "# samples_size = " << samples_size << endl;
  HSSOptions<double> hss_opts;
  hss_opts.set_verbose(true);
  hss_opts.set_from_command_line(argc, argv);

  vector<double> data_train       = write_from_file(samples_size, filename + "_train.csv");
  vector<double> data_test        = write_from_file(samples_size, filename + "_test.csv");
  vector<double> data_train_label = write_from_file(samples_size, filename + "_train_label.csv");
  vector<double> data_test_label  = write_from_file(samples_size, filename + "_test_label.csv");

  int n = data_train.size() / d;
  int m = data_test.size() / d;
  cout << "# matrix size = " << n << " x " << d << endl;

  HSSPartitionTree cluster_tree;
  cluster_tree.size = n;
  int cluster_size = hss_opts.leaf_size();
  if (reorder == 1) {
    cout << " 2 means" << endl;
    recursive_2_means(data_train.data(), n, d, cluster_size, cluster_tree,
                      data_train_label.data());
  } else if (reorder == 2) {
    recursive_kd(data_train.data(), n, d, cluster_size, cluster_tree,
                 data_train_label.data());
    cout << "kd " << endl;
  }

  cout << "constructing Kdense .. " << endl;

  DenseMatrix<double> Kdense(n, n);
  if (kernel == 1) {
    for (int c = 0; c < n; c++)
      for (int r = 0; r < n; r++)
        Kdense(r, c) =
            Gauss_kernel(&data_train[r * d], &data_train[c * d], d, h);
  } else {
    for (int c = 0; c < n; c++)
      for (int r = 0; r < n; r++)
        Kdense(r, c) =
            Laplace_kernel(&data_train[r * d], &data_train[c * d], d, h);
  }
  if (lambda != 0) {
    cout << " adding lambda " << lambda << endl;
    for (int i = 0; i < n; i++) Kdense(i, i) += lambda;
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
  K.compress(Kdense, hss_opts);
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
  cout << "# memory(K) = " << K.memory() / 1e6 << " MB, "
       << 100. * K.memory() / Kdense.memory() << "% of dense" << endl;

  // solve test

  timer.start();
  auto ULV = K.factor();
  cout << "# factor time = " << timer.elapsed() << endl;

  DenseMatrix<double> B(n, 1, &data_train_label[0], n);
  DenseMatrix<double> weights(B);

  timer.start();
  K.solve(ULV, weights);
  cout << "# solve time = " << timer.elapsed() << endl;

  
  auto Bcheck = K.apply(weights);

  Bcheck.scaled_add(-1., B);
  cout << "# relative error = ||B-H*(H\\B)||_F/||B||_F = "
       << Bcheck.normF() / B.normF() << endl;

  double* prediction = new double[m];
  for (int i = 0; i < m; ++i) {
    prediction[i] = 0;
  }

  if (kernel == 1) {
    for (int c = 0; c < m; c++)
      for (int r = 0; r < n; r++)
        prediction[c] +=
            Gauss_kernel(&data_train[r * d], &data_test[c * d], d, h) *
            weights(r, 0);
  } else {
    for (int c = 0; c < m; c++)
      for (int r = 0; r < n; r++)
        prediction[c] +=
            Laplace_kernel(&data_train[r * d], &data_test[c * d], d, h) *
            weights(r, 0);
  }

  for (int i = 0; i < m; ++i) {
    prediction[i] = ((prediction[i] > 0) ? 1. : -1.);
  }
  // compute accuracy score of prediction
  double incorrect_quant = 0;
  for (int i = 0; i < m; ++i) {
    double a = (prediction[i] - data_test_label[i]) / 2;
    incorrect_quant += (a > 0 ? a : -a);
  }
  cout << "prediction score: " << ((m - incorrect_quant) / m) * 100 << "%"
       << endl;

  return 0;
}

#include <cmath>
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include <vector>

namespace py = pybind11;

// std::vector<int> pb_divide(int length, int block_size, int n_blocks) {
//     int l_total = block_size * n_blocks;
//     double overlap = 1.0 * (l_total - length) / (n_blocks - 1) / 2;
//     double l_minus_edges = 1.0 * length - 2 * overlap;
//     std::vector<int> lim_a(n_blocks);
//     // std::vector<int> lim_b(n_blocks);

//     double halfblock = 0.5 * block_size;
//     for (int i = 0; i < n_blocks; i++) {
//         double center_point = overlap + l_minus_edges * (i + 0.5) / (n_blocks);
//         lim_a[i] = lrint(center_point - halfblock);
//         // lim_b[i] = round(center_point + halfblock);
//     }
//     return lim_a;
// }

// template <typename T>
// py::array_t<T, py::array::c_style | py::array::forcecast>
// pb_combine_volume(py::array_t<T, py::array::c_style | py::array::forcecast> blocklist,
//                   std::vector<int> shape, double sampling) {

//     auto data = blocklist.unchecked<4>();
//     int n_blocks = static_cast<int>(data.shape(0));
//     int b0 = static_cast<int>(data.shape(1));
//     int b1 = static_cast<int>(data.shape(2));
//     int b2 = static_cast<int>(data.shape(3));

//     int nx = shape[0];
//     int ny = shape[1];
//     int nz = shape[2];
//     //   py::print("shape", nx, ny, nz);

//     py::array_t<T> retval;
//     py::array_t<T> norm;
//     retval.resize({nx, ny, nz});
//     norm.resize({nx, ny, nz});
//     std::fill(retval.mutable_data(), retval.mutable_data() + retval.size(), 0.);
//     std::fill(norm.mutable_data(), norm.mutable_data() + norm.size(), 0.);

//     auto data_im = retval.mutable_unchecked<3>();
//     auto data_norm = norm.mutable_unchecked<3>();

//     int ndiv0 = int(ceil(sampling * nx / b0));
//     int ndiv1 = int(ceil(sampling * ny / b1));
//     int ndiv2 = int(ceil(sampling * nz / b2));

//     std::vector<int> b0x = pb_divide(nx, b0, ndiv0);
//     std::vector<int> b0y = pb_divide(ny, b1, ndiv1);
//     std::vector<int> b0z = pb_divide(nz, b2, ndiv2);

//     int block_index = 0;
//     for (int i = 0; i < ndiv0; i++) {
//         int block_pos_x = b0x[i];
//         for (int j = 0; j < ndiv1; j++) {
//             int block_pos_j = b0y[j];
//             for (int k = 0; k < ndiv2; k++) {
//                 int block_pos_k = b0z[k];
//                 for (int bi = 0; bi < b0; bi++) {
//                     for (int bj = 0; bj < b1; bj++) {
//                         for (int bk = 0; bk < b2; bk++) {
//                             data_im(block_pos_x + bi, block_pos_j + bj, block_pos_k + bk) +=
//                                 data(block_index, bi, bj, bk);
//                             data_norm(block_pos_x + bi, block_pos_j + bj, block_pos_k + bk)
//                             += 1.0;
//                         }
//                     }
//                 }
//                 block_index++;
//             }
//         }
//     }

//     for (ssize_t i = 0; i < data_im.shape(0); i++)
//         for (ssize_t j = 0; j < data_im.shape(1); j++)
//             for (ssize_t k = 0; k < data_im.shape(2); k++)
//                 data_im(i, j, k) /= data_norm(i, j, k);

//     return retval;
// }

template <typename T>
py::array_t<T> bin_volume(py::array_t<T, py::array::c_style | py::array::forcecast> volume,
                          int binning) {
    auto data = volume.unchecked<3>();
    int nx = static_cast<int>(data.shape(0));
    int ny = static_cast<int>(data.shape(1));
    int nz = static_cast<int>(data.shape(2));

    int nx_binned = nx / binning;
    int ny_binned = ny / binning;
    int nz_binned = nz / binning;

    py::array_t<T> retval;
    retval.resize({nx_binned, ny_binned, nz_binned});
    std::fill(retval.mutable_data(), retval.mutable_data() + retval.size(), 0.);
    auto data_out = retval.mutable_unchecked<3>();

    double scale = 1.0 / (binning * binning * binning);
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                int ib = i / binning;
                int jb = j / binning;
                int kb = k / binning;
                data_out(ib, jb, kb) += scale * data(i, j, k);
            }
        }
    }
    return retval;
}

template <typename T>
py::array_t<T> bin_tensor(py::array_t<T, py::array::c_style | py::array::forcecast> tensor,
                          int binning) {
    auto data = tensor.unchecked<4>();
    int channels = static_cast<int>(data.shape(0));
    int nx = static_cast<int>(data.shape(1));
    int ny = static_cast<int>(data.shape(2));
    int nz = static_cast<int>(data.shape(3));

    int nx_binned = nx / binning;
    int ny_binned = ny / binning;
    int nz_binned = nz / binning;

    py::array_t<T> retval;
    retval.resize({channels, nx_binned, ny_binned, nz_binned});
    std::fill(retval.mutable_data(), retval.mutable_data() + retval.size(), 0.);
    auto data_out = retval.mutable_unchecked<4>();

    double scale = 1.0 / (binning * binning * binning);
    for (int ch = 0; ch < channels; ch++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    int ib = i / binning;
                    int jb = j / binning;
                    int kb = k / binning;
                    data_out(ch, ib, jb, kb) += scale * data(ch, i, j, k);
                }
            }
        }
    }
    return retval;
}

template <typename T>
py::array_t<T> upscale_volume(py::array_t<T, py::array::c_style | py::array::forcecast> volume,
                              int binning) {
    auto data = volume.unchecked<3>();
    int nx = static_cast<int>(data.shape(0));
    int ny = static_cast<int>(data.shape(1));
    int nz = static_cast<int>(data.shape(2));

    int nx_scaled = nx * binning;
    int ny_scaled = ny * binning;
    int nz_scaled = nz * binning;

    py::array_t<T> retval;
    retval.resize({nx_scaled, ny_scaled, nz_scaled});
    std::fill(retval.mutable_data(), retval.mutable_data() + retval.size(), 0.);
    auto data_out = retval.mutable_unchecked<3>();

    for (int is = 0; is < nx_scaled; is++) {
        for (int js = 0; js < ny_scaled; js++) {
            for (int ks = 0; ks < nz_scaled; ks++) {
                int i = is / binning;
                int j = js / binning;
                int k = ks / binning;
                data_out(is, js, ks) += data(i, j, k);
            }
        }
    }
    return retval;
}

template <typename T>
py::array_t<T> upscale_tensor(py::array_t<T, py::array::c_style | py::array::forcecast> tensor,
                              int binning) {
    auto data = tensor.unchecked<4>();
    int channels = static_cast<int>(data.shape(0));
    int nx = static_cast<int>(data.shape(1));
    int ny = static_cast<int>(data.shape(2));
    int nz = static_cast<int>(data.shape(3));

    int nx_scaled = nx * binning;
    int ny_scaled = ny * binning;
    int nz_scaled = nz * binning;

    py::array_t<T> retval;
    retval.resize({channels, nx_scaled, ny_scaled, nz_scaled});
    std::fill(retval.mutable_data(), retval.mutable_data() + retval.size(), 0.);
    auto data_out = retval.mutable_unchecked<4>();

    for (int ch = 0; ch < channels; ch++) {
        for (int is = 0; is < nx_scaled; is++) {
            for (int js = 0; js < ny_scaled; js++) {
                for (int ks = 0; ks < nz_scaled; ks++) {
                    int i = is / binning;
                    int j = js / binning;
                    int k = ks / binning;
                    data_out(ch, is, js, ks) += data(ch, i, j, k);
                }
            }
        }
    }
    return retval;
}

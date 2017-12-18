rm -rf build
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=. \
-DCMAKE_C_COMPILER=mpicc \
-DCMAKE_CXX_COMPILER=mpic++ \
-DBLAS_LIBRARIES=/usr/local/Cellar/openblas/0.2.20/lib/libblas.dylib \
-DLAPACK_LIBRARIES=/usr/local/Cellar/openblas/0.2.20/lib/liblapack.dylib \
-DSCALAPACK_LIBRARIES="/usr/local/Cellar/scalapack/2.0.2_8/lib/libscalapack.dylib" \
-DCMAKE_CXX_FLAGS="" \
-DCMAKE_Fortran_COMPILER=mpifort \
-DMETIS_INCLUDES=/usr/local/Cellar/metis/5.1.0/include \
-DMETIS_LIBRARIES=/usr/local/Cellar/metis/5.1.0/lib/libmetis.dylib \
-DPARMETIS_INCLUDES=/usr/local/Cellar/parmetis/4.0.3_4/include \
-DPARMETIS_LIBRARIES=/usr/local/Cellar/parmetis/4.0.3_4/lib/libparmetis.dylib \
-DSCOTCH_INCLUDES=/usr/local/Cellar/scotch/6.0.4_4/include \
-DSCOTCH_LIBRARIES="/usr/local/Cellar/scotch/6.0.4_4/lib/libscotch.dylib;/usr/local/Cellar/scotch/6.0.4_4/lib/libscotcherr.dylib;/usr/local/Cellar/scotch/6.0.4_4/lib/libptscotch.dylib;/usr/local/Cellar/scotch/6.0.4_4/lib/libptscotcherr.dylib"

# -DCMAKE_CXX_FLAGS="-lstdc++ -DUSE_TASK_TIMER -DCOUNT_FLOPS" \

make install VERBOSE=1
cd examples
make -j 4
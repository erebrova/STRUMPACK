#!/bin/bash

if [[ $2 == "r" ]];
then
rm -rf build
mkdir build
fi

cd build

if [[ $1 == "cori" ]];
then
echo $1
#source ../configEnv.sh
export CRAYPE_LINK_TYPE="dynamic"
export PARMETIS_INSTALL="/global/cscratch1/sd/gichavez/edison/intel17/parmetis-4.0.3"
export SCOTCH_INSTALL="/global/cscratch1/sd/gichavez/edison/intel17/scotch_6.0.4/build"
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=. \
-DCMAKE_CXX_COMPILER=CC \
-DCMAKE_C_COMPILER=cc \
-DCMAKE_Fortran_COMPILER=ftn \
-DCMAKE_EXE_LINKER_FLAGS="" \
-DCMAKE_CXX_FLAGS="" \
-DMETIS_INCLUDES=$PARMETIS_INSTALL/metis/include \
-DMETIS_LIBRARIES=$PARMETIS_INSTALL/build/Linux-x86_64/libmetis/libmetis.a \
-DPARMETIS_INCLUDES=$PARMETIS_INSTALL/install/include \
-DPARMETIS_LIBRARIES=$PARMETIS_INSTALL/install/lib/libparmetis.a \
-DHMATRIX_LIBRARIES=/global/cscratch1/sd/gichavez/intel17/h_matrix_rbf_randomization/build/SRC/libhmatrix.a \
-DSCOTCH_INCLUDES=$SCOTCH_INSTALL/include \
-DSCOTCH_LIBRARIES="$SCOTCH_INSTALL/lib/libscotch.a;$SCOTCH_INSTALL/lib/libscotcherr.a;$SCOTCH_INSTALL/lib/libptscotch.a;$SCOTCH_INSTALL/lib/libptscotcherr.a"
elif [[ $1 == "edison" ]];
then
echo $1
source ../configEnv.sh
export CRAYPE_LINK_TYPE="dynamic"
export PARMETIS_INSTALL="/global/cscratch1/sd/gichavez/intel17/parmetis-4.0.3"
export SCOTCH_INSTALL="/global/cscratch1/sd/gichavez/intel17/scotch_6.0.4/build"
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=. \
-DCMAKE_CXX_COMPILER=CC \
-DCMAKE_C_COMPILER=cc \
-DCMAKE_Fortran_COMPILER=ftn \
-DCMAKE_EXE_LINKER_FLAGS="" \
-DCMAKE_CXX_FLAGS="" \
-DMETIS_INCLUDES=$PARMETIS_INSTALL/metis/include \
-DMETIS_LIBRARIES=$PARMETIS_INSTALL/build/Linux-x86_64/libmetis/libmetis.a \
-DPARMETIS_INCLUDES=$PARMETIS_INSTALL/install/include \
-DPARMETIS_LIBRARIES=$PARMETIS_INSTALL/install/lib/libparmetis.a \
-DSCOTCH_INCLUDES=$SCOTCH_INSTALL/include \
-DSCOTCH_LIBRARIES="$SCOTCH_INSTALL/lib/libscotch.a;$SCOTCH_INSTALL/lib/libscotcherr.a;$SCOTCH_INSTALL/lib/libptscotch.a;$SCOTCH_INSTALL/lib/libptscotcherr.a"
elif [[ $1 == "imac" ]];
then
echo $1
cmake .. -DCMAKE_BUILD_TYPE=Debug \
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
else
	echo "Unrecognized configuration. Try: <cori|edison|imac>"
	exit 0
fi

# -DCMAKE_BUILD_TYPE=Debug
# -DCMAKE_CXX_FLAGS="-lstdc++ -DUSE_TASK_TIMER -DCOUNT_FLOPS" \

make install VERBOSE=1
cd examples
make

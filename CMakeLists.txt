cmake_minimum_required(VERSION 3.7)
project(impulse_vectorized)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetBuilder/BaseBuilder.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetBuilder/BaseBuilder.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetBuilder/CSVBuilder.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetBuilder/CSVBuilder.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/Callback.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/Callback.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/Category.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/Category.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/MinMaxScaling.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/MinMaxScaling.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/MissingData.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/MissingData.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/ZScoresScaling.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/ZScoresScaling.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/BaseDatasetModifier.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/BaseDatasetModifier.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSlicer.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSlicer.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSplitter.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSplitter.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetSample.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetSample.h
        src/Vendor/json.hpp
        main.cpp
        src/Impulse/NeuralNetwork/Builder/Builder.h
        src/Impulse/NeuralNetwork/Network.h
        src/Impulse/NeuralNetwork/NetworkSerializer.h
        src/Impulse/NeuralNetwork/Layer/Logistic.h
        src/Impulse/NeuralNetwork/Layer/Relu.h
        src/Impulse/NeuralNetwork/Layer/Abstract.h
        src/Impulse/NeuralNetwork/Trainer/AbstractTrainer.h
        src/Impulse/NeuralNetwork/Trainer/AbstractTrainer.cpp
        src/Impulse/NeuralNetwork/Math/common.h
        src/Impulse/NeuralNetwork/Math/Fmincg.cpp
        src/Impulse/NeuralNetwork/Math/Fmincg.h
        src/Impulse/common.h
        src/Impulse/NeuralNetwork/Trainer/common.h
        src/Impulse/NeuralNetwork/Trainer/CojungateGradientTrainer.cpp
        src/Impulse/NeuralNetwork/Trainer/CojungateGradientTrainer.h
        src/Impulse/NeuralNetwork/Math/common.cpp)

find_package(Boost 1.57.0 COMPONENTS system filesystem REQUIRED)
if(BOOST_FOUND)
    message("Boost FOUND")
    include_directories(${Boost_INCLUDE_DIRS})
endif()

#OpenMP
#-fopenmp
FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#Intel MLK
#[[include(cmake/FindMKL.cmake)
if(MKL_FOUND)
    include_directories(MLK_INCLUDE_DIRS)
    link_directories(MKL_LIBRARIES)
endif()]]

#-lblas
#[[FIND_PACKAGE(BLAS REQUIRED)
if(BLAS_FOUND)
    message("BLAS FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${BLAS_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BLAS_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_EXE_LINKER_FLAGS}")
endif()]]

#its slower with lapack
#-llapack
#[[FIND_PACKAGE(LAPACK REQUIRED)
if(LAPACK_FOUND)
    message("LAPACK FOUND")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")
    link_directories(LAPACK_LIBRARIES)
endif()]]

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -m64 -lboost_system -lboost_filesystem -flto -march=native -funroll-loops -mavx -msse4.2 -fopenmp")

add_executable(impulse_vectorized ${SOURCE_FILES})

target_link_libraries(impulse_vectorized ${Boost_LIBRARIES})
#target_link_libraries(impulse_vectorized ${BLAS_LIBRARIES})

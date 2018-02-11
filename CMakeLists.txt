cmake_minimum_required(VERSION 3.7)
project(impulse_vectorized)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/common.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/common.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetBuilder/Abstract.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetBuilder/Abstract.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetBuilder/CSVBuilder.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetBuilder/CSVBuilder.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/Callback.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/Callback.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/Category.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/Category.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/CategoryId.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/CategoryId.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/MinMaxScaling.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/MinMaxScaling.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/MissingData.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/MissingData.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/ZScoresScaling.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/ZScoresScaling.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/Abstract.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/Modifier/Abstract.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/DatasetSlicer.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/DatasetSlicer.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/DatasetSplitter.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetModifier/DatasetSplitter.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/Dataset.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/Dataset.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetSample.cpp
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/DatasetSample.h
        src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/include.h
        main.cpp
        src/Vendor/json/src/json.hpp
        src/Impulse/NeuralNetwork/common.h
        src/Impulse/NeuralNetwork/Builder/Abstract.h
        src/Impulse/NeuralNetwork/Builder/Abstract.cpp
        src/Impulse/NeuralNetwork/Builder/ClassifierBuilder.h
        src/Impulse/NeuralNetwork/Builder/ClassifierBuilder.cpp
        src/Impulse/NeuralNetwork/Builder/ConvBuilder.h
        src/Impulse/NeuralNetwork/Builder/ConvBuilder.cpp
        src/Impulse/NeuralNetwork/Network/Abstract.h
        src/Impulse/NeuralNetwork/Network/ConvNetwork.h
        src/Impulse/NeuralNetwork/Network/ConvNetwork.cpp
        src/Impulse/NeuralNetwork/Network/ClassifierNetwork.h
        src/Impulse/NeuralNetwork/Network/ClassifierNetwork.cpp
        src/Impulse/NeuralNetwork/Serializer.h
        src/Impulse/NeuralNetwork/Layer/Logistic.h
        src/Impulse/NeuralNetwork/Layer/Relu.h
        src/Impulse/NeuralNetwork/Layer/Abstract.h
        src/Impulse/NeuralNetwork/Trainer/Abstract.h
        src/Impulse/NeuralNetwork/Trainer/Abstract.cpp
        src/Impulse/NeuralNetwork/Math/common.h
        src/Impulse/NeuralNetwork/Math/Fmincg.cpp
        src/Impulse/NeuralNetwork/Math/Fmincg.h
        src/Impulse/NeuralNetwork/Trainer/common.h
        src/Impulse/NeuralNetwork/Trainer/ConjugateGradient.cpp
        src/Impulse/NeuralNetwork/Trainer/ConjugateGradient.h
        src/Impulse/NeuralNetwork/Trainer/GradientDescent.cpp
        src/Impulse/NeuralNetwork/Trainer/GradientDescent.h
        src/Impulse/NeuralNetwork/Trainer/MiniBatchGradientDescent.cpp
        src/Impulse/NeuralNetwork/Trainer/MiniBatchGradientDescent.h
        src/Impulse/NeuralNetwork/Math/common.cpp
        src/Impulse/NeuralNetwork/Network/Abstract.cpp
        src/Impulse/NeuralNetwork/Serializer.cpp
        src/Impulse/NeuralNetwork/Layer/Abstract.cpp
        src/Impulse/NeuralNetwork/Layer/Logistic.cpp
        src/Impulse/NeuralNetwork/Layer/Relu.cpp
        src/Impulse/NeuralNetwork/Layer/Softmax.h
        src/Impulse/NeuralNetwork/Layer/Softmax.cpp
        src/Impulse/NeuralNetwork/Layer/Purelin.h
        src/Impulse/NeuralNetwork/Layer/Purelin.cpp
        src/Impulse/NeuralNetwork/Layer/Conv.h
        src/Impulse/NeuralNetwork/Layer/Conv.cpp
        src/Impulse/NeuralNetwork/Layer/MaxPool.h
        src/Impulse/NeuralNetwork/Layer/MaxPool.cpp
        src/Impulse/NeuralNetwork/Layer/FullyConnected.h
        src/Impulse/NeuralNetwork/Layer/FullyConnected.cpp
        src/Impulse/NeuralNetwork/include.h
        src/Impulse/NeuralNetwork/utils.h
        src/Impulse/NeuralNetwork/utils.cpp
        src/Impulse/NeuralNetwork/Layer/Abstract1D.h
        src/Impulse/NeuralNetwork/Layer/Abstract1D.cpp
        src/Impulse/NeuralNetwork/Layer/Abstract2D.h
        src/Impulse/NeuralNetwork/Layer/Abstract2D.cpp
        src/Impulse/NeuralNetwork/Layer/BackPropagation/Factory.cpp
        src/Impulse/NeuralNetwork/Layer/BackPropagation/Factory.h
        src/Impulse/NeuralNetwork/Layer/BackPropagation/Abstract.h
        src/Impulse/NeuralNetwork/Layer/BackPropagation/Abstract.cpp
        src/Impulse/NeuralNetwork/Layer/BackPropagation/BackPropagation1DTo1D.h
        src/Impulse/NeuralNetwork/Layer/BackPropagation/BackPropagation1DTo1D.cpp
        src/Impulse/NeuralNetwork/Layer/BackPropagation/BackPropagationToMaxPool.h
        src/Impulse/NeuralNetwork/Layer/BackPropagation/BackPropagationToMaxPool.cpp
        src/Impulse/NeuralNetwork/Layer/BackPropagation/BackPropagationToConv.cpp
        src/Impulse/NeuralNetwork/Layer/BackPropagation/BackPropagationToConv.h
        src/Impulse/NeuralNetwork/Layer/BackPropagation/BackPropagation2DTo1D.h
        src/Impulse/NeuralNetwork/Layer/BackPropagation/BackPropagation2DTo1D.cpp src/Impulse/NeuralNetwork/Math/Matrix.h src/Impulse/NeuralNetwork/Math/Matrix.cpp)

find_package(Boost 1.57.0 COMPONENTS system filesystem REQUIRED)
if (BOOST_FOUND)
    message("Boost FOUND")
    include_directories(${Boost_INCLUDE_DIRS})
endif ()

#Intel MLK
#[[find_package(MKL REQUIRED)
if(MKL_FOUND)
    message("MKL FOUND")
    add_definitions(-DENABLE_MKL)
    add_definitions(-DMKL_LP64)
    add_definitions(-DEIGEN_BLAS_INDEX=int)
endif()]]

#OpenMP
#-fopenmp
FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    include_directories(/opt/intel/mkl/include)
endif ()

#-lblas
FIND_PACKAGE(BLAS REQUIRED)
if(BLAS_FOUND)
    message("BLAS FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${BLAS_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BLAS_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_EXE_LINKER_FLAGS}")
endif()

#its slower with lapack
#-llapack
#[[FIND_PACKAGE(LAPACK REQUIRED)
if(LAPACK_FOUND)
    message("LAPACK FOUND")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")
    link_directories(LAPACK_LIBRARIES)
endif()]]

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -m64 -flto -march=native -funroll-loops -mavx -msse4.2 -fopenmp")

add_executable(impulse_vectorized ${SOURCE_FILES})

target_link_libraries(impulse_vectorized ${Boost_LIBRARIES})
target_link_libraries(impulse_vectorized ${OpenCV_LIBS})
target_link_libraries(impulse_vectorized ${BLAS_LIBRARIES})
#target_link_libraries(impulse_vectorized ${MKL_LIBRARIES})
cmake_minimum_required(VERSION 3.7)
project(impulse_ml_dataset)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/src/Impulse/DatasetBuilder/BaseBuilder.cpp
        src/src/Impulse/DatasetBuilder/BaseBuilder.h
        src/src/Impulse/DatasetBuilder/CSVBuilder.cpp
        src/src/Impulse/DatasetBuilder/CSVBuilder.h
        src/src/Impulse/DatasetModifier/Modifier/Callback.cpp
        src/src/Impulse/DatasetModifier/Modifier/Callback.h
        src/src/Impulse/DatasetModifier/Modifier/Category.cpp
        src/src/Impulse/DatasetModifier/Modifier/Category.h
        src/src/Impulse/DatasetModifier/Modifier/MinMaxScaling.cpp
        src/src/Impulse/DatasetModifier/Modifier/MinMaxScaling.h
        src/src/Impulse/DatasetModifier/Modifier/MissingData.cpp
        src/src/Impulse/DatasetModifier/Modifier/MissingData.h
        src/src/Impulse/DatasetModifier/Modifier/ZScoresScaling.cpp
        src/src/Impulse/DatasetModifier/Modifier/ZScoresScaling.h
        src/src/Impulse/DatasetModifier/BaseDatasetModifier.cpp
        src/src/Impulse/DatasetModifier/BaseDatasetModifier.h
        src/src/Impulse/DatasetModifier/DatasetSlicer.cpp
        src/src/Impulse/DatasetModifier/DatasetSlicer.h
        src/src/Impulse/DatasetModifier/DatasetSplitter.cpp
        src/src/Impulse/DatasetModifier/DatasetSplitter.h
        src/src/Impulse/Dataset.cpp
        src/src/Impulse/Dataset.h
        src/src/Impulse/DatasetSample.cpp
        src/src/Impulse/DatasetSample.h
        )

add_executable(impulse_ml_dataset ${SOURCE_FILES})
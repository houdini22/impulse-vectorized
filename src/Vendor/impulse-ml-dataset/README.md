# impulse-ml

## impulse-ml-dataset

```cpp
#include "src/Impulse/Dataset.h"
#include "src/Impulse/DatasetBuilder/CSVBuilder.h"
#include "src/Impulse/DatasetModifier/Modifier/Callback.h"
#include "src/Impulse/DatasetModifier/Modifier/Category.h"
#include "src/Impulse/DatasetModifier/Modifier/MissingData.h"
#include "src/Impulse/DatasetModifier/DatasetSlicer.h"
#include "src/Impulse/DatasetModifier/DatasetSplitter.h"
#include "src/Impulse/DatasetModifier/Modifier/ZScoresScaling.h"

int main() {
	// build dataset from csv...
	Impulse::DatasetBuilder::CSVBuilder builder(
			"/home/hud/eclipse-workspace/impulse-ml-dataset/src/data/data.csv");
	Impulse::Dataset dataset = builder.build();
	dataset.out();

	// and start modification - slice to input-output pair
	Impulse::DatasetSlicer datasetSlicer(&dataset);
	datasetSlicer.addInputColumn(0);
	datasetSlicer.addInputColumn(1);
	datasetSlicer.addInputColumn(2);
	datasetSlicer.addOutputColumn(3);

	Impulse::SlicedDataset slicedDataset = datasetSlicer.slice();
	slicedDataset.input.out();
	slicedDataset.output.out();

	// fill missing data by mean of fields
	Impulse::DatasetModifier::Modifier::MissingData missingDataModifier(
			&slicedDataset.input);
	missingDataModifier.setModificationType("mean");
	missingDataModifier.applyToColumn(1);
	missingDataModifier.applyToColumn(2);

	slicedDataset.input.out();

	Impulse::DatasetModifier::Modifier::Category categoryModifier(
			&slicedDataset.input);
	categoryModifier.applyToColumn(0);

	slicedDataset.input.out();

	// convert last column "Yes/No" to numbers
	Impulse::DatasetModifier::Modifier::Callback callbackModifier(
			&slicedDataset.output);
	callbackModifier.setCallback([](std::string oldValue) -> std::string {
		if(oldValue.find("Yes") != std::string::npos) {
			return "1";
		}
		return "0";
	});
	callbackModifier.applyToColumn(0);

	slicedDataset.output.out();

	// scale all input
	Impulse::DatasetModifier::Modifier::ZScoresScaling zScoresModifier(
			&slicedDataset.input);

	zScoresModifier.applyToAll();

	// we can split dataset by parts of given ratio
	Impulse::DatasetSplitter datasetSplitter(&slicedDataset);
	Impulse::SplittedDataset splittedDataset = datasetSplitter.split(0.8);

	splittedDataset.primary.input.out();
	splittedDataset.primary.output.out();
	splittedDataset.secondary.input.out();
	splittedDataset.secondary.output.out();

	return 0;
}
```
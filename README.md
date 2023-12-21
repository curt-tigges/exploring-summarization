# exploring-summarization

## Datasets
The main class for toy patching datasets is `summarization_utils.toy_datasets::CounterfactualDataset()`.

### Creating datasets
To create a dataset, you have three options:
1. Provide prompts and answers as lists of strings to `CounterfactualDataset.__init__()`, e.g. 
```
dataset = CounterfactualDataset(
    prompts=["Known for being the first to walk on the moon, Neil", ...],
    answers=[" Armstrong", ...],
    cf_prompts=["Known for being the star of the movie Jazz Singer, Neil", ...],
    cf_answers=[" Diamond", ...],
)
```
2. Provide prompts and answers as a list of tuples to `CounterfactualDataset.from_tuples()`, e.g.
```
dataset = CounterfactualDataset.from_tuples([
    (
        "Known for being the first to walk on the moon, Neil",
        " Armstrong",
        "Known for being the star of the movie Jazz Singer, Neil",
        " Diamond",
    ),
    ...
])
```
3. If you want to generate a dataset using a template like `"{NAME} is {ATTR1}. {NAME} is {ATTR2}. {NAME} is {ATTR3}. Is {NAME} {ATTR_R}?"`, you can create a subclass of the abstract base class `summarization_utils.toy_datasets::TemplaticDataset()`. The abstract methods which must be defined in the sub-class are 
* `get_counterfactual_tuples()` - determines how the prompts are altered to create the counterfactual prompts `cf_prompts`
* `get_answers` - determines the solution to the algorthmic task, generating `answers`
* `format_prompts` - determines how the template is formatted from a list of variables.

Finally, you can call the `to_counterfactual()` method to obtain a `CounterfactualDataset` as above. For an example, see the `BooleanNegatorDataset`.

For all three cases, you can also add a case to `CounterfactualDataset.from_name` for easier access once your dataset is fairly stable.

### Loading datasets
Here is how to easily load 3 hard-coded datasets and 5 templatic datasets into a single object:
```
names = ["KnownFor", "OfCourse", "Code", "BooleanNegator", "BooleanOperator", "ToyBinding", "ToyDeduction", "ToyProfiles"]
dataset = sum([
    CounterfactualDataset.from_name(name, model) for name in names
], start=CounterfactualDataset.empty(model))
```
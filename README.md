# LLMDataPrepEnv

An agentic environment for autonomous data cleaning and formatting, building on the OpenEnv specification. 
Designed for evaluating LLMs on their data wrangling and cleaning capabilities in a programmatic API loop.

## Motivation & Real-world Utility
Data cleaning is an integral task for Data Scientists and Engineers. Messy datasets containing null values, conflicting data types, and non-standardized strings represent real friction. `LLMDataPrepEnv` models this reality by dropping an agent into a dirty pipeline (e.g., mixing date formats like `MM/DD/YYYY` and currencies like `200 EUR`) and challenging it to mutate the dataset until it perfectly aligns with a target schema.

## Observation Space
The `Observation` model exposes:
- `dataset`: List of row objects (e.g., representing Users / Purchases).
- `target_schema`: Target typing rules mapping fields to validations (e.g., `signup_date` must be ISO YYYY-MM-DD).
- `conversion_rates`: Mappings allowing agents to appropriately convert strings like `100 EUR` to USD floats.

## Action Space
1. `DeleteRowAction(row_index, reason)`
2. `UpdateValueAction(row_index, column, new_value)`
3. `PassAction()`

## Tasks & Grading Logic
This environment incorporates deterministic evaluators ranging in difficulty:
- **Easy (Null Purge)**: Grader returns 1.0 if all rows featuring `null` emails are successfully removed.
- **Medium (Date Standardization)**: Grader returns 1.0 if `signup_date` strings strictly match the ISO 8601 format (`YYYY-MM-DD`). 
- **Hard (Currency Conversion)**: Grader verifies that `purchase_total` strings are correctly stripped of symbols and accurately converted into USD `float` types relying on the context provided in `conversion_rates`.

## Usage Instructions

```bash
# Optional: Install dependencies
pip install pydantic openai openenv

# Run the baseline execution
python inference.py
```

Expected output runs through a strict stdout format (`[START]`, `[STEP]`, `[END]`) which enables automated OpenEnv evaluations.

To run via Docker (e.g., for HuggingFace Spaces):
```bash
docker build -t openenv-dataprep .
docker run -e OPENAI_API_KEY=your_key openenv-dataprep
```

# Code by Azadeh Nia, November 2025

# AzureOpenAI DocProcessing Usage Guide
This solution has been created to use generative AI and azure services to extract structured data in Markdown format from unstructured documents (pdf),
then store the vectorized data and structured data in respective databases for client applications such as Microsoft Copilot Studio.

If you intend to use these tools in production make sure you properly test, review an adjust the code.

## 0. Providing the environment variables
check the env_sample file, rename it to .env and provide all the required values.

## 1. Setting Up a Python Virtual Environment

It is recommended to use a virtual environment to manage dependencies and avoid conflicts.


### On Windows (PowerShell)

#python --version: 3.13.9

1. Open PowerShell in the project directory (or terminal in VSCode)
2. Run:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```
3. Install required packages:
   ```powershell
   pip install -r requirements.txt
   ```

## 2. How to Use the Code

- Place your input files (PDFs, DOCXs, CSVs, etc.) in the `data/input/` folder.
- The main script is `info_extractor.py`.
- To run the script:
  ```powershell
  python info_extractor.py
  ```
- The script will process the files and extract information as per the logic defined.
- In the main() section, adjust any input and output file names.

## 3. Customization
- provide environment variables in the env file
- You can modify these Prompts to adjust extraction instructions.

## 4. Troubleshooting
- Ensure your virtual environment is activated before running the script.
- If you add new dependencies, update `requirements.txt` and re-run `pip install -r requirements.txt`.

## 5. Additional Notes
- In this solution gpt-4.1 model has been used with the chat completion endpoint
- To produce structured output in json or other formats, you can use the md output and add another step with generative AI to provide the desired format and fields.
- Please check the input document sensitivity labels and protection rights, this code will not work on the sensitive documents.
- For questions or issues, review the code comments in `info_extractor.py`.

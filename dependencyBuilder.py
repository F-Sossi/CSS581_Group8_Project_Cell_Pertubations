import ast
import os
import json

def extract_imports(code):
    tree = ast.parse(code)
    return [node.names[0].name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

def get_libraries_from_file(file_path):
    with open(file_path, 'r') as file:
        if file_path.endswith('.ipynb'):
            content = json.load(file)
            code_cells = [cell['source'] for cell in content['cells'] if cell['cell_type'] == 'code']
            code = '\n'.join(['\n'.join(cell) for cell in code_cells])
        else:
            code = file.read()
    return extract_imports(code)

def create_toml_file(libraries, file_path='pyproject.toml'):
    with open(file_path, 'w') as file:
        file.write('[tool.poetry.dependencies]\n')
        file.write('python = "^3.8"\n')
        for library in libraries:
            file.write(f'{library} = "*"\n')

def setup_libraries(file_paths):
    libraries = set()
    for file_path in file_paths:
        libraries.update(get_libraries_from_file(file_path))
    create_toml_file(libraries)

ipynb_files = ['Data_exploration.ipynb', 'tensorflow.ipynb', 'XGboost.ipynb']
setup_libraries(ipynb_files)